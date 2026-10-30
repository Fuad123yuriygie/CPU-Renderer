#include "bsdf.h"

namespace cr {
// Fresnel reflectance averages the two light-polarization components. Equal
// indices remove the interface; sinT2 >= 1 is the total-internal-reflection case.
float fresnelDielectric(float cosine, float etaI, float etaT) {
    if (etaI==etaT) return 0;
    cosine=std::clamp(std::abs(cosine),0.0f,1.0f);
    const float eta=etaI/etaT;
    const float sinT2=eta*eta*std::max(0.0f,1-cosine*cosine);
    if (sinT2>=1) return 1;
    const float cosT=std::sqrt(1-sinT2);
    const float ds=etaI*cosine+etaT*cosT, dp=etaT*cosine+etaI*cosT;
    if (ds==0 || dp==0) return 1;
    const float rs=(etaI*cosine-etaT*cosT)/ds, rp=(etaT*cosine-etaI*cosT)/dp;
    return 0.5f*(rs*rs+rp*rp);
}
// Refract the reversed outgoing vector through a microfacet normal. Keeping
// failure explicit lets the sampler select reflection instead of producing NaNs.
bool refractDirection(Vec3 wo, Vec3 normal, float ratio, Vec3& wi) {
    const float cosI=dot(wo,normal);
    const float sinT2=ratio*ratio*std::max(0.0f,1-cosI*cosI);
    if (sinT2>=1) return false;
    wi=normalize(-wo*ratio+normal*(ratio*cosI-std::sqrt(1-sinT2)));
    return true;
}
// glTF's attenuation color describes transmission after attenuationDistance.
// Solving color = exp(-sigma * distance) gives sigma = -log(color) / distance.
Vec3 absorptionCoefficient(Vec3 color, float distance) {
    if (!std::isfinite(distance)) return {};
    if (!(distance>0)) throw std::runtime_error("Attenuation distance must be positive");
    color=clamp(color,1e-8f,1);
    return {-std::log(color.x)/distance,-std::log(color.y)/distance,-std::log(color.z)/distance};
}
// Absorption is applied to actual geometric segment lengths, not material
// thickness. The infinity branch avoids the undefined product 0 * infinity.
Vec3 beerLambert(Vec3 coefficient, float distance) {
    if (distance==infinity) return {coefficient.x>0?0.0f:1.0f,coefficient.y>0?0.0f:1.0f,coefficient.z>0?0.0f:1.0f};
    return exp(coefficient*(-std::max(0.0f,distance)));
}
namespace {
// GGX normal-distribution function D: alpha is squared perceptual roughness,
// and h is a unit microfacet normal in the local shading frame (+Z is up).
float distribution(float alpha, Vec3 h) {
    if (h.z<=0) return 0;
    const float a2=alpha*alpha;
    const float d=(h.x*h.x+h.y*h.y)+a2*h.z*h.z;
    return a2/(pi*d*d);
}
// Smith's one-direction visibility factor models microfacets masking each other.
// Using squared z makes the same function usable on either transmission side.
float smithG1(float alpha, Vec3 v) {
    const float cos2=v.z*v.z;
    if (cos2==0) return 0;
    return 2/(1+std::sqrt(1+alpha*alpha*(v.x*v.x+v.y*v.y)/cos2));
}
// Heitz's visible-normal GGX sampler avoids wasting almost all grazing samples.
Vec3 visibleNormal(Vec3 wo, float alpha, float u, float v) {
    // Stretch the view so the anisotropy caused by roughness becomes a hemisphere.
    const Vec3 vh=normalize({alpha*wo.x,alpha*wo.y,wo.z});
    const float lensq=vh.x*vh.x+vh.y*vh.y;
    const Vec3 t1=lensq>0 ? Vec3{-vh.y,vh.x,0}/std::sqrt(lensq) : Vec3{1,0,0};
    const Vec3 t2=cross(vh,t1);
    // Warp a uniform disk sample to the visible portion of that hemisphere.
    const float r=std::sqrt(u), phi=2*pi*v;
    const float x=r*std::cos(phi);
    float y=r*std::sin(phi);
    const float s=0.5f*(1+vh.z);
    y=(1-s)*std::sqrt(std::max(0.0f,1-x*x))+s*y;
    const Vec3 nh=t1*x+t2*y+vh*std::sqrt(std::max(0.0f,1-x*x-y*y));
    // Undo the stretch to obtain a GGX-distributed normal visible from wo.
    return normalize({alpha*nh.x,alpha*nh.y,std::max(0.0f,nh.z)});
}
// Per-scattering setup shared by evaluation and sampling. The shading frame is
// oriented toward wo, but Surface keeps its outward geometric normal unchanged.
// Sharing these probabilities is essential: the sampler and PDF must agree.
struct Context {
    const Surface& s;
    Frame frame;
    Vec3 wo;
    float etaI, etaT, eta, alpha, diffuseProbability, deltaTransmissionProbability=0;
    bool thin, smooth, matched=false;
    // Reverse the dielectric IOR only at a closed-volume exit. An opaque
    // two-sided backface must not accidentally behave like the inside of glass.
    Context(const Surface& surface, Vec3 outgoing)
        : s(surface),frame(dot(outgoing,s.normal)>0?s.normal:-s.normal),wo(frame.local(outgoing)),
          etaI(s.thickness>0&&!s.frontFace?s.ior:1),etaT(s.thickness>0&&!s.frontFace?1:s.ior),eta(etaT/etaI),
          alpha(std::max(0.001f,s.roughness*s.roughness)),diffuseProbability(0),
          thin(s.thickness<=0 && s.transmission>0),smooth(s.roughness<=0.001f) {
        if (thin) { etaI=1; etaT=s.ior; eta=s.ior; }
        // Transmission half-vector inversion amplifies float direction error by
        // 1/|eta-1|. Below sqrt(epsilon), adopt the matched-index limit rather
        // than resolve that ill-conditioned lobe. The relative test is symmetric.
        matched=std::abs(etaI-etaT)<=std::sqrt(std::numeric_limits<float>::epsilon())*std::max(etaI,etaT);
        if (matched) etaI=etaT=eta=1;
        // These luminance estimates choose lobes efficiently; dividing by their
        // probabilities later preserves each lobe's actual RGB contribution.
        const float f=fresnel(wo.z);
        const float diffuse=(1-s.metallic)*(1-s.transmission)*luminance(s.baseColor)*(1-f);
        const float transmit=(1-s.metallic)*s.transmission*(1-f)*luminance(s.baseColor);
        const float total=diffuse+luminance(reflection(f,wo.z))+transmit;
        if (total>0) {
            diffuseProbability=diffuse/total;
            if (matched) deltaTransmissionProbability=transmit/total;
        }
    }
    // Whatever probability remains belongs to GGX reflection/refraction.
    float microProbability() const { return std::max(0.0f,1-diffuseProbability-deltaTransmissionProbability); }
    // A parallel, zero-thickness sheet combines the two dielectric interfaces.
    float fresnel(float cosine) const {
        const float f=fresnelDielectric(cosine,etaI,etaT);
        return thin ? 2*f/(1+f) : f;
    }
    // Dielectrics use scalar Fresnel; metals use colored Schlick reflectance.
    Vec3 reflection(float f, float cosine) const {
        const float schlick=std::pow(1-std::clamp(std::abs(cosine),0.0f,1.0f),5.0f);
        return Vec3{(1-s.metallic)*f}+(s.baseColor+(Vec3{1}-s.baseColor)*schlick)*s.metallic;
    }
    // Only the nonmetal portion transmits, and reflected energy is removed first.
    Vec3 transmission(float f) const { return s.baseColor*((1-s.metallic)*s.transmission*(1-f)); }
    // This is conditional on selecting the microfacet lobe and its normal.
    // Matched-index transmission was already separated into its own delta lobe.
    float reflectionProbability(float f, float cosH) const {
        if (matched) return 1;
        const float r=luminance(reflection(f,cosH)), t=luminance(transmission(f));
        return r+t>0?r/(r+t):1;
    }
};
// Evaluate f and the full sampling PDF in local coordinates. Positive wi.z is
// reflection; negative wi.z is transmission. Delta distributions cannot be
// evaluated as finite directional densities, so their continuous value is zero.
BsdfEval evalLocal(const Context& c, Vec3 wi) {
    BsdfEval result;
    const float no=c.wo.z, ni=std::abs(wi.z);
    if (no<=1e-7f || ni<=1e-7f) return result;
    const bool reflection=wi.z>0;
    if (!reflection && (c.matched || c.thin)) return result;
    if (reflection) {
        // Lambertian reflection is reduced by metal/transmission weights and
        // interface Fresnel factors, rather than added as an independent energy source.
        const float diffuse=(1-c.s.metallic)*(1-c.s.transmission)*(1-c.fresnel(no))*(1-c.fresnel(ni));
        result.value=c.s.baseColor*(diffuse/pi);
        result.pdf=c.diffuseProbability*ni/pi;
    }
    if (c.smooth) return result;
    // Reflection bisects wo and wi. Refraction requires an IOR-weighted half-vector.
    Vec3 h=normalize(reflection ? c.wo+wi : c.wo+wi*c.eta);
    if (h.z<0) h=-h;
    const float oh=dot(c.wo,h), ih=dot(wi,h);
    if (oh<=0 || (reflection?ih<=0:ih>=0)) return result;
    const float f=c.fresnel(oh), pr=c.reflectionProbability(f,oh);
    const float d=distribution(c.alpha,h), g1=smithG1(c.alpha,c.wo), g=g1*smithG1(c.alpha,wi);
    // Convert visible-normal probability into direction probability with the
    // reflection/refraction Jacobian; include every lobe-selection probability.
    const float pdfH=d*g1*oh/no;
    if (reflection) {
        result.value+=c.reflection(f,oh)*(d*g/(4*no*ni));
        result.pdf+=c.microProbability()*pr*pdfH/(4*oh);
    } else {
        const float denom=oh+c.eta*ih, denom2=denom*denom;
        if (denom2<1e-20f) return {};
        // Radiance transport includes (etaI/etaT)^2; it cancels the eta^2 in the BTDF Jacobian.
        result.value=c.transmission(f)*(d*g*std::abs(oh*ih)/(no*ni*denom2));
        result.pdf=c.microProbability()*(1-pr)*pdfH*c.eta*c.eta*std::abs(ih)/denom2;
    }
    return result;
}
// A normal map may bend the shading frame, but it cannot make reflection pass
// through the real triangle or make transmission stay on the incident side.
bool correctGeometricSide(const Surface& s, Vec3 wo, Vec3 wi, bool transmitted) {
    const float d=dot(wo,s.geometricNormal)*dot(wi,s.geometricNormal);
    return transmitted ? d<0 : d>0;
}
}
// Public evaluation adapts world directions to the shared local-frame evaluator.
BsdfEval evaluateBsdf(const Surface& s, Vec3 wo, Vec3 wi) {
    Context c(s,wo);
    const Vec3 local=c.frame.local(wi);
    if (!correctGeometricSide(s,wo,wi,local.z<0)) return {};
    return evalLocal(c,local);
}
// Select a lobe, generate its direction, then return the complete mixture's
// f * abs(cos(theta)) / PDF. This is the path-throughput update used by trace().
BsdfSample sampleBsdf(const Surface& s, Vec3 wo, Rng& rng) {
    Context c(s,wo);
    BsdfSample out;
    if (c.wo.z<=1e-7f) return out;
    Vec3 wi;
    const float lobe=rng.uniform();
    if (lobe<c.diffuseProbability) {
        // A cosine-weighted hemisphere matches the Lambertian cosine factor.
        const float u=rng.uniform(), phi=2*pi*rng.uniform();
        wi={std::sqrt(u)*std::cos(phi),std::sqrt(u)*std::sin(phi),std::sqrt(1-u)};
    } else if (lobe<c.diffuseProbability+c.deltaTransmissionProbability) {
        // Equal indices erase the interface, regardless of microfacet roughness.
        // Keep this delta separate so a mixed metal's reflection stays rough.
        out.direction=-wo; out.delta=true; out.transmitted=true; out.pdf=c.deltaTransmissionProbability;
        out.weight=s.baseColor*((1-s.metallic)*s.transmission/out.pdf);
        return correctGeometricSide(s,wo,out.direction,true)?out:BsdfSample{};
    } else {
        // Rough surfaces sample a visible microfacet; perfectly smooth ones
        // use the shading normal itself. Fresnel then chooses reflection/transmission.
        const float u=rng.uniform(), v=rng.uniform();
        const Vec3 h=c.smooth ? Vec3{0,0,1} : visibleNormal(c.wo,c.alpha,u,v);
        const float oh=dot(c.wo,h), f=c.fresnel(oh), pr=c.reflectionProbability(f,oh);
        const bool reflection=rng.uniform()<pr;
        if (reflection) wi=reflect(-c.wo,h);
        else if (c.thin) wi=-c.wo;
        else if (!refractDirection(c.wo,h,1/c.eta,wi)) return out;
        if ((reflection && wi.z<=0) || (!reflection && wi.z>=0)) return out;
        out.transmitted=!reflection;
        out.eta=out.transmitted&&!c.thin?c.eta:1;
        if (c.smooth || (!reflection && c.thin)) {
            // Delta weights use event probabilities, not a solid-angle PDF.
            // Radiance crossing a volume also carries the inverse-squared IOR ratio.
            out.delta=true;
            out.direction=normalize(c.frame.world(wi));
            if (!correctGeometricSide(s,wo,out.direction,out.transmitted)) return {};
            const float prob=c.microProbability()*(reflection?pr:1-pr);
            if (prob<=0) return {};
            out.pdf=prob;
            out.weight=(reflection?c.reflection(f,oh):c.transmission(f)/(out.eta*out.eta))/prob;
            return out;
        }
    }
    // Evaluate the exact world direction being returned, after normalization.
    // Reusing an earlier local PDF can be inaccurate for a very sharp BTDF.
    out.direction=normalize(c.frame.world(wi));
    wi=c.frame.local(out.direction);
    if ((wi.z<0)!=out.transmitted || !correctGeometricSide(s,wo,out.direction,out.transmitted)) return {};
    const auto e=evalLocal(c,wi);
    if (e.pdf<=0 || !std::isfinite(e.pdf)) return {};
    out.pdf=e.pdf;
    // Return the estimator weight once, so the integrator cannot divide twice.
    out.weight=e.value*(std::abs(wi.z)/e.pdf);
    return out;
}
}
