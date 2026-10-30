// Deterministic transport tests: analytic optics, independent Monte Carlo
// integration, MIS energy accounting, small rendered fixtures, and worker parity.
#include "render.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

using namespace cr;
namespace {
// Exceptions, unlike assert(), stay active when Release defines NDEBUG.
void require(bool value,const std::string& text) { if (!value) throw std::runtime_error(text); }
// Scalar comparisons carry an explicit tolerance and report both values on failure.
void near(float a,float b,float tolerance,const std::string& text) {
    if (std::abs(a-b)>tolerance) throw std::runtime_error(text+": "+std::to_string(a)+" vs "+std::to_string(b));
}
// An independent uniform-direction proposal for integrating the BSDF/PDF. It
// deliberately does not reuse the renderer's importance-sampling implementation.
Vec3 uniformSphere(Rng& rng) {
    const float z=1-2*rng.uniform(), phi=2*pi*rng.uniform(), r=std::sqrt(1-z*z);
    return {r*std::cos(phi),r*std::sin(phi),z};
}
// Start each material test from a known outward frame and nonmetal base material.
Surface surface() {
    Surface s;
    s.normal=s.geometricNormal={0,0,1}; s.tangent={1,0,0}; s.bitangent={0,1,0};
    s.baseColor=Vec3{0.6f}; s.roughness=0.65f; s.metallic=0; s.ior=1.5f; s.frontFace=true;
    return s;
}
// Closed-form checks isolate optics from geometry: 4% normal-incidence glass
// reflection, total internal reflection, Snell direction, and known absorption.
void testFresnelAndAbsorption() {
    near(fresnelDielectric(1,1,1.5f),0.04f,1e-6f,"normal Fresnel");
    near(fresnelDielectric(0.5f,1.5f,1),1,1e-6f,"total internal reflection Fresnel");
    Vec3 wi;
    require(!refractDirection(normalize({0.9f,0,0.3f}),{0,0,1},1.5f,wi),"TIR must not refract");
    require(refractDirection({0,0,1},{0,0,1},1/1.5f,wi),"normal refraction");
    near(wi.z,-1,1e-6f,"normal transmitted direction");
    const Vec3 coefficient=absorptionCoefficient({0.25f,0.5f,1},2);
    const Vec3 two=beerLambert(coefficient,2), four=beerLambert(coefficient,4);
    near(two.x,0.25f,1e-6f,"known attenuation red"); near(two.y,0.5f,1e-6f,"known attenuation green");
    near(four.x,0.0625f,1e-6f,"attenuation doubles optical path"); near(four.z,1,1e-6f,"unabsorbed blue");
    require(finite(beerLambert(coefficient,infinity)),"infinite segment finite attenuation");
    Surface s=surface(); s.transmission=1; s.thickness=1; s.roughness=0; s.baseColor=Vec3{1};
    // With white smooth glass, branch frequency should be Fresnel reflectance,
    // and transmitted radiance must include the inverse-squared IOR ratio.
    Rng rng(9); int reflected=0, transmitted=0;
    for (int i=0;i<50000;++i) {
        const auto b=sampleBsdf(s,{0,0,1},rng); require(static_cast<bool>(b)&&b.delta,"smooth dielectric delta sample");
        if (b.transmitted) { ++transmitted; near(b.weight.x,1/2.25f,2e-5f,"radiance refraction eta correction"); }
        else { ++reflected; near(b.weight.x,1,2e-5f,"Fresnel reflection weight"); }
    }
    near(static_cast<float>(reflected)/static_cast<float>(reflected+transmitted),0.04f,0.004f,"Fresnel branch frequency");
    s.frontFace=false;
    for (int i=0;i<500;++i) {
        const auto b=sampleBsdf(s,normalize({0.9f,0,-0.3f}),rng);
        require(static_cast<bool>(b)&&!b.transmitted,"TIR sample always reflects");
    }
    s.frontFace=true; s.thickness=0;
    // A parallel thin sheet transmits straight through even at oblique incidence.
    const Vec3 wo=normalize({0.4f,0,1});
    bool found=false;
    for (int i=0;i<30;++i) {
        const auto b=sampleBsdf(s,wo,rng);
        if (b.transmitted) { require(length(b.direction+wo)<1e-5f,"thin sheet straight-through transmission"); near(b.eta,1,0,"thin sheet no volume IOR tracking"); found=true; }
    }
    require(found,"thin sheet transmitted samples exist");
    // Backfaces of opaque two-sided surfaces must not acquire glass-exit TIR.
    s=surface(); s.doubleSided=true;
    const Vec3 frontWo=normalize({0.8f,0,0.5f}), frontWi=normalize({-0.3f,0.1f,0.8f});
    const auto front=evaluateBsdf(s,frontWo,frontWi);
    s.frontFace=false;
    const auto back=evaluateBsdf(s,{frontWo.x,frontWo.y,-frontWo.z},{frontWi.x,frontWi.y,-frontWi.z});
    require(length(front.value-back.value)<1e-5f,"opaque backface retains the same dielectric Fresnel");
    near(front.pdf,back.pdf,1e-5f,"opaque double-sided PDF parity");
    // Equal-index media remove the interface, but mixing in a metal must still
    // retain that metal's rough continuous reflection lobe.
    s=surface(); s.ior=1; s.transmission=1; s.thickness=1; s.baseColor=Vec3{1}; s.roughness=0.7f;
    for (int i=0;i<1000;++i) {
        const auto b=sampleBsdf(s,wo,rng);
        require(b&&b.delta&&b.transmitted,"rough equal-index transmission is a delta");
        require(length(b.direction+wo)<1e-6f&&length(b.weight-Vec3{1})<1e-6f,"equal-index straight-through energy");
    }
    s.metallic=0.35f;
    bool roughReflection=false, straightTransmission=false;
    for (int i=0;i<1000;++i) {
        const auto b=sampleBsdf(s,wo,rng);
        if (b.transmitted) straightTransmission|=b.delta;
        else if (b) roughReflection|=!b.delta;
    }
    require(roughReflection&&straightTransmission,"equal-index mixture preserves rough metal and delta transmission lobes");
}
// Check both local sample/evaluation agreement and independent spherical
// integration, so matching bugs in the sampler and its PDF cannot validate each other.
void testBsdf() {
    const Vec3 wo=normalize({0.3f,0.1f,1});
    for (int kind=0;kind<4;++kind) {
        // Cases: diffuse dielectric, metal, rough closed dielectric, and thin sheet.
        Surface s=surface();
        if (kind==1) { s.metallic=1; s.baseColor={0.8f,0.45f,0.15f}; s.roughness=0.5f; }
        if (kind>=2) { s.transmission=0.85f; s.thickness=kind==2?1.0f:0.0f; s.baseColor=Vec3{0.92f}; s.roughness=0.7f; s.ior=1.333f; }
        Rng rng(72+static_cast<uint64_t>(kind)), uniform(819+static_cast<uint64_t>(kind));
        Vec3 sampled{}, integrated{}, totalEnergy{};
        double pdfIntegral=0; int accepted=0;
        constexpr int count=180000;
        for (int i=0;i<count;++i) {
            const auto b=sampleBsdf(s,wo,rng);
            if (b) {
                totalEnergy+=b.weight;
                // Uniform directions cannot integrate a delta distribution.
                // Count delta energy separately from the continuous comparison.
                if (b.delta) require(kind==3&&b.transmitted&&length(b.direction+wo)<1e-5f,"rough thin sheet transmits straight through");
                else {
                    ++accepted;
                    require(b.pdf>0,"rough sample has continuous PDF");
                    const auto e=evaluateBsdf(s,wo,b.direction);
                    require(finite(e.value)&&e.pdf>=0,"finite nonnegative BSDF");
                    near(e.pdf,b.pdf,1e-4f*std::max(1.0f,b.pdf),"sample/eval PDF parity");
                    const Vec3 weight=e.value*(std::abs(dot(s.normal,b.direction))/b.pdf);
                    require(length(weight-b.weight)<0.003f*std::max(1.0f,length(weight)),"f*cos/pdf sampling weight");
                    sampled+=b.weight;
                }
            }
            const Vec3 wi=uniformSphere(uniform);
            const auto e=evaluateBsdf(s,wo,wi);
            // Uniform sphere PDF is 1/(4*pi); multiplying by 4*pi undoes it.
            integrated+=e.value*(4*pi*std::abs(wi.z)); pdfIntegral+=e.pdf*4*pi;
        }
        sampled=sampled/static_cast<float>(count); integrated=integrated/static_cast<float>(count);
        const float relative=length(sampled-integrated)/std::max(0.05f,length(sampled));
        require(relative<0.055f,"independent uniform-vs-BSDF integration, kind "+std::to_string(kind)+" relative "+std::to_string(relative));
        // Rejected microfacet/geometric-side samples have zero contribution.
        // The remaining continuous PDF integrates to their acceptance probability.
        near(static_cast<float>(pdfIntegral/count),static_cast<float>(accepted)/count,0.025f,"PDF integral equals non-null sample probability");
        require(maxComponent(totalEnergy/static_cast<float>(count))<=1.03f,"BSDF must not create energy");
    }
}
// Stress the ill-conditioned IOR-near-one limit, including grazing rays,
// opposite interface sides, mixed lobes, and a transformed world normal frame.
void testNearIndexBsdf() {
    Mat4 transform;
    transform.m[0]=-2; transform.m[5]=0.4f; transform.m[10]=3;
    transform.m[4]=0.3f; transform.m[8]=0.7f;
    const Vec3 normals[]{Vec3{0,0,1},transform.normal(normalize({1,2,3}))};
    const float indices[]{1,std::nextafter(1.0f,2.0f),1.000001f,1.0001f,1.01f,1.333f,1.5f};
    Rng rng(0x902fde3f3ULL);
    for (Vec3 normal:normals) for (bool front:{true,false}) for (float ior:indices)
    for (float roughness:{0.03f,0.4f,0.85f}) for (float cosine:{0.01f,0.5f,0.8f}) for (bool mixed:{false,true}) {
        Surface s=surface();
        s.normal=s.geometricNormal=normal; s.frontFace=front;
        s.ior=ior; s.roughness=roughness; s.thickness=1;
        s.metallic=mixed?0.35f:0; s.transmission=mixed?0.6f:1;
        s.baseColor=mixed?Vec3{0.8f,0.45f,0.15f}:Vec3{1};
        const Frame frame(front?normal:-normal);
        const float sine=std::sqrt(1-cosine*cosine);
        const Vec3 wo=normalize(frame.world({0.8f*sine,0.6f*sine,cosine}));
        const bool matched=ior<=1.0001f;
        int transmitted=0, continuous=0;
        for (int i=0;i<512;++i) {
            const auto b=sampleBsdf(s,wo,rng);
            require(finite(b.direction)&&finite(b.weight)&&std::isfinite(b.pdf),"near-index finite sample");
            if (matched&&!mixed) require(b&&b.delta&&b.transmitted,"matched pure transmission must not lose samples");
            if (!b) continue;
            near(length(b.direction),1,2e-6f,"near-index unit world direction");
            require(std::min({b.weight.x,b.weight.y,b.weight.z})>=0,"near-index nonnegative weight");
            const auto e=evaluateBsdf(s,wo,b.direction);
            require(finite(e.value)&&std::isfinite(e.pdf),"near-index finite returned-direction evaluation");
            if (b.transmitted) ++transmitted;
            if (b.delta) {
                // Matched-index transmission must remain unit-energy and absent
                // from the continuous PDF, not a narrow lobe with a singular half-vector.
                require(matched&&b.transmitted,"only matched transmission may be delta in a rough material");
                require(length(b.direction+wo)<1e-6f&&b.eta==1,"matched limit is straight-through with unit eta");
                require(e.pdf==0&&maxComponent(e.value)==0,"matched delta excluded from continuous eval/PDF");
                const Vec3 amplitude=s.baseColor*((1-s.metallic)*s.transmission);
                require(length(b.weight*b.pdf-amplitude)<2e-6f,"matched delta lobe weight and probability");
                if (!mixed) require(length(b.weight-Vec3{1})<1e-6f,"matched pure transmission preserves furnace energy");
            } else {
                // Evaluate the returned WORLD direction, not a hidden intermediate.
                ++continuous;
                require(e.pdf>0&&e.pdf==b.pdf,"continuous PDF must match the returned world direction exactly");
                const Vec3 weight=e.value*(std::abs(dot(s.normal,b.direction))/e.pdf);
                require(length(weight-b.weight)<2e-6f*std::max(1.0f,length(weight)),"returned-direction f*cos/pdf weight");
                if (matched) {
                    Surface reference=s; reference.ior=1;
                    const auto exact=evaluateBsdf(reference,wo,b.direction);
                    require(length(exact.value-e.value)<1e-6f&&exact.pdf==e.pdf,"matched limit preserves diffuse and rough metal lobes");
                }
            }
        }
        if (matched) {
            require(transmitted>0,"matched transmission lobe remains sampleable");
            if (mixed) require(continuous>0,"matched mixture must retain continuous reflection");
        } else require(continuous>0,"ordinary indices must retain rough continuous scattering");
    }
}
// A bright asymmetric HDR texel exercises importance selection and spherical
// area weighting. Uniform integration checks normalization independently.
void testEnvironment() {
    Image im; im.width=16; im.height=8; im.pixels.resize(128,{0.2f,0.2f,0.2f,1});
    im.pixels[37]={10,4,1,1};
    Environment env(im,31,2);
    Rng rng(781), uniform(25);
    double integral=0;
    for (int i=0;i<40000;++i) {
        float pdf=0; const Vec3 d=env.sample(rng,pdf);
        near(length(d),1,1e-5f,"environment unit sample");
        near(pdf,env.pdf(d),std::max(0.00002f,0.0001f*pdf),"environment sample/pdf reciprocity");
        require(pdf>0&&finite(env.radiance(d)),"environment finite positive density");
        integral+=env.pdf(uniformSphere(uniform))*4*pi;
    }
    near(static_cast<float>(integral/40000),1,0.04f,"environment spherical PDF normalization");
    // Rotating lookup and rotating a known direction must agree at the texture seam.
    Environment unrotated(im,0,2), rotated(im,90,2);
    const Vec3 a=unrotated.radiance({1,0,0}), b=rotated.radiance({0,0,1});
    require(length(a-b)<1e-5f,"environment rotation parity");
    Environment constant(Vec3{0.3f});
    near(static_cast<float>(constant.integral()),0.3f*4*pi,1e-5f,"constant environment integral");
}
// Small in-memory test geometry: split a quad while preserving outward winding,
// flat normals, and a simple UV square. No demo-scene construction lives here.
void quad(Scene& s,Vec3 a,Vec3 b,Vec3 c,Vec3 d,uint32_t mat) {
    Triangle t; t.material=mat;
    const Vec3 n=normalize(cross(b-a,c-a));
    t.v[0].position=a; t.v[1].position=b; t.v[2].position=c;
    t.v[0].uv[0]={0,0}; t.v[1].uv[0]={1,0}; t.v[2].uv[0]={1,1};
    for (auto& v:t.v) v.normal=n;
    s.triangles.push_back(t);
    t.v[1].position=c; t.v[2].position=d; t.v[1].uv[0]={1,1}; t.v[2].uv[0]={0,1};
    s.triangles.push_back(t);
}
// A closed, outward cube makes entry/exit medium tracking testable from every face.
void cube(Scene& s,uint32_t m) {
    quad(s,{-0.7f,-0.7f,0.7f},{0.7f,-0.7f,0.7f},{0.7f,0.7f,0.7f},{-0.7f,0.7f,0.7f},m);
    quad(s,{0.7f,-0.7f,-0.7f},{-0.7f,-0.7f,-0.7f},{-0.7f,0.7f,-0.7f},{0.7f,0.7f,-0.7f},m);
    quad(s,{-0.7f,-0.7f,-0.7f},{-0.7f,-0.7f,0.7f},{-0.7f,0.7f,0.7f},{-0.7f,0.7f,-0.7f},m);
    quad(s,{0.7f,-0.7f,0.7f},{0.7f,-0.7f,-0.7f},{0.7f,0.7f,-0.7f},{0.7f,0.7f,0.7f},m);
    quad(s,{-0.7f,0.7f,0.7f},{0.7f,0.7f,0.7f},{0.7f,0.7f,-0.7f},{-0.7f,0.7f,-0.7f},m);
    quad(s,{-0.7f,-0.7f,-0.7f},{0.7f,-0.7f,-0.7f},{0.7f,-0.7f,0.7f},{-0.7f,-0.7f,0.7f},m);
}
// Fixed viewpoint and linear mean keep the following energy comparisons repeatable.
Camera camera() { Camera c; c.name="Test"; c.position={0,0,4}; c.forward={0,0,-1}; c.yfov=0.65f; return c; }
float mean(const RenderResult& image) { double total=0; for (Vec3 p:image.pixels) total+=luminance(p); return static_cast<float>(total/image.pixels.size()); }
// A diffuse plane under a unit sky should return its albedo. Adding a separate
// emitter then checks that direct sampling and emitter hits do not double energy.
void testEnergyAndMis() {
    Scene s; Material m; m.metallic=0; m.baseColor={0.6f,0.6f,0.6f,1}; m.ior=1; s.materials.push_back(m);
    quad(s,{-20,-20,0},{20,-20,0},{20,20,0},{-20,20,0},0); s.finalize();
    Bvh bvh; bvh.build(s); Environment env(Vec3{1});
    RenderOptions o; o.width=16; o.height=16; o.samples=512; o.threads=2; o.bounces=2; o.russianRoulette=false;
    const auto mis=render(s,bvh,camera(),env,o);
    o.nextEventEstimation=false;
    const auto plain=render(s,bvh,camera(),env,o);
    near(mean(plain),0.6f,1e-4f,"constant sky diffuse energy");
    near(mean(mis),mean(plain),0.012f,"environment MIS must not double emission");
    Material emitter; emitter.baseColor={0,0,0,1}; emitter.metallic=0; emitter.ior=1; emitter.emission={2,1.5f,1};
    s.materials.push_back(emitter);
    quad(s,{-2,-2,2},{-2,2,2},{2,2,2},{2,-2,2},1); s.finalize(); bvh.build(s);
    Camera c=camera(); c.position={0,0,1}; c.yfov=0.45f;
    o.samples=1024;
    const auto lightPlain=render(s,bvh,c,Environment{},o);
    o.nextEventEstimation=true;
    const auto lightMis=render(s,bvh,c,Environment{},o);
    require(mean(lightMis)>0.3f,"emissive area lights illuminate receiver");
    near(mean(lightMis),mean(lightPlain),0.018f,"triangle emission/MIS accounting");
    // Primary camera visibility is not a competing light-sampling technique.
    c.forward={0,0,1}; c.right={-1,0,0};
    o.samples=4;
    const auto visible=render(s,bvh,c,Environment{},o);
    near(visible.pixels[0].x,2,1e-5f,"camera directly sees unweighted emission");
}
// Render five compact feature scenes twice with identical sample keys. A change
// in worker count may change scheduling, but must not change any accumulated pixel.
void testFixturesAndWorkers() {
    std::filesystem::create_directories("test-renders");
    const char* names[]{"diffuse","metal","water","foliage","lit-room"};
    for (int kind=0;kind<5;++kind) {
        Scene s; Material ground; ground.metallic=0; ground.baseColor={0.4f,0.35f,0.25f,1}; s.materials.push_back(ground);
        Material object; object.metallic=0; object.baseColor={0.55f,0.16f,0.08f,1}; object.roughness=0.4f;
        if (kind==1) { object.metallic=1; object.baseColor={0.9f,0.55f,0.16f,1}; object.roughness=0.2f; }
        if (kind==2) { object.transmission=1; object.thickness=1; object.baseColor={1,1,1,1}; object.attenuationColor={0.25f,0.85f,0.7f}; object.attenuationDistance=2; object.ior=1.333f; object.roughness=0.14f; }
        if (kind==3) {
            // Alternating transparent texels make a genuine alpha cutout, not BLEND.
            object.doubleSided=true; object.alphaMode=AlphaMode::Mask; object.baseColor={0.12f,0.45f,0.04f,1};
            Image im; im.width=2; im.height=2; im.pixels={{1,1,1,1},{1,1,1,0},{1,1,1,0},{1,1,1,1}};
            s.images.push_back(im); Texture tex; tex.image=0; tex.linear=false; s.textures.push_back(tex); object.baseColorTexture.texture=0;
        }
        s.materials.push_back(object);
        quad(s,{-8,-0.8f,8},{8,-0.8f,8},{8,-0.8f,-8},{-8,-0.8f,-8},0);
        if (kind==3) quad(s,{-0.8f,-0.7f,0},{0.8f,-0.7f,0},{0.8f,0.9f,0},{-0.8f,0.9f,0},1);
        else cube(s,1);
        if (kind==4) {
            quad(s,{-2,-0.8f,-2},{2,-0.8f,-2},{2,3,-2},{-2,3,-2},0);
            Material emission; emission.emission=Vec3{5}; emission.metallic=0; emission.baseColor={0,0,0,1}; s.materials.push_back(emission);
            quad(s,{-0.8f,2,-0.8f},{0.8f,2,-0.8f},{0.8f,2,0.8f},{-0.8f,2,0.8f},2);
        } else { PunctualLight light; light.type=LightType::Directional; light.direction=normalize({-1,-2,-1}); light.intensity=2; s.lights.push_back(light); }
        s.finalize(); Bvh bvh; bvh.build(s);
        RenderOptions o; o.width=32; o.height=32; o.samples=32; o.bounces=8; o.threads=1;
        Environment env(Vec3{kind==4?0.01f:0.3f});
        const auto one=render(s,bvh,camera(),env,o); o.threads=4;
        const auto many=render(s,bvh,camera(),env,o);
        require(one.pixels.size()==many.pixels.size(),"worker image dimensions");
        for (std::size_t i=0;i<one.pixels.size();++i) {
            const Vec3 a=one.pixels[i], b=many.pixels[i];
            require(finite(a)&&a.x>=0&&a.y>=0&&a.z>=0,"finite nonnegative fixture");
            require(a.x==b.x&&a.y==b.y&&a.z==b.z,"bit-identical one/four-worker accumulation");
        }
        require(mean(one)>0.01f,"fixture not blank");
        // Keep tiny PNG/HDR artifacts for visual inspection and cross-build metrics.
        const std::filesystem::path path=std::filesystem::path("test-renders")/names[kind];
        io::writePng(path.string()+".png",o.width,o.height,one.pixels,0);
        io::writeHdr(path.string()+".hdr",o.width,o.height,one.pixels);
    }
}
// Automatic rendering should reserve CPU headroom, while explicit user counts
// remain honored. A constant empty scene makes this scheduling check inexpensive.
void testWorkerHeadroom() {
    Scene scene; Bvh bvh; bvh.build(scene);
    RenderOptions options; options.width=options.height=64; options.samples=1; options.bounces=1;
    const Environment environment(Vec3{0.1f});
    const auto automatic=render(scene,bvh,camera(),environment,options);
    const int expected=std::min(16,static_cast<int>(std::max(1u,std::thread::hardware_concurrency()/2)));
    require(automatic.workers==expected,"automatic rendering reserves half the logical CPUs");
    options.threads=2;
    const auto explicitCount=render(scene,bvh,camera(),environment,options);
    require(explicitCount.workers==2,"explicit worker override is honored");
    for(std::size_t i=0;i<automatic.pixels.size();++i)
        require(lengthSquared(automatic.pixels[i]-explicitCount.pixels[i])==0,"CPU headroom policy preserves deterministic pixels");
}
}
int main() {
    try {
        // Run lightweight analytic checks before the rendered integration fixtures.
        testFresnelAndAbsorption(); std::cout << "PASS Fresnel, TIR, thin sheets, and known-path absorption\n";
        testBsdf(); std::cout << "PASS independent BSDF energy/PDF integration\n";
        testNearIndexBsdf(); std::cout << "PASS near-index limits and returned-direction BSDF/PDF consistency\n";
        testEnvironment(); std::cout << "PASS HDR solid-angle sampling and rotation\n";
        testEnergyAndMis(); std::cout << "PASS constant environment and emissive/MIS accounting\n";
        testFixturesAndWorkers(); std::cout << "PASS five fixtures and bit-identical worker results\n";
        testWorkerHeadroom(); std::cout << "PASS automatic CPU headroom and explicit worker override\n";
        // Bad dimensions must fail validation before film allocation or tracing.
        RenderOptions invalid; invalid.width=0; bool rejected=false;
        try { invalid.validate(); } catch (const std::exception&) { rejected=true; }
        require(rejected,"invalid dimensions rejected");
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
