#include "render.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>

namespace cr {
// A constant environment has equal radiance in every direction. Its integral
// over the sphere is 4*pi times luminance; black is a valid, unlit default.
Environment::Environment(Vec3 constant) : constant_(constant),integral_(4*pi*luminance(constant)) {
    if (!finite(constant) || std::min({constant.x,constant.y,constant.z})<0) throw std::runtime_error("Invalid environment radiance");
}
// Exact solid angle of one latitude/longitude texel. Polar rows cover less of
// the sphere than equatorial rows and must not receive equal sampling weight.
double Environment::solidAngle(int row) const {
    const double a=static_cast<double>(pi)*row/image_.height, b=static_cast<double>(pi)*(row+1)/image_.height;
    return (2*static_cast<double>(pi)/image_.width)*(std::cos(a)-std::cos(b));
}
// Build a reusable, immutable importance distribution from linear HDR pixels.
// Texture brightness affects how often a direction is sampled, not its radiance.
Environment::Environment(Image image, float rotationDegrees, float intensity)
    : image_(std::move(image)),rotation_(std::fmod(rotationDegrees,360.0f)*pi/180),intensity_(intensity) {
    if (image_.width<=0 || image_.height<=0 || image_.pixels.size()!=static_cast<std::size_t>(image_.width)*image_.height)
        throw std::runtime_error("Invalid environment image dimensions");
    if (!std::isfinite(rotationDegrees)||!std::isfinite(intensity)||intensity<0) throw std::runtime_error("Invalid environment rotation/intensity");
    cdf_.reserve(image_.pixels.size());
    for (int y=0;y<image_.height;++y) {
        const double omega=solidAngle(y);
        for (int x=0;x<image_.width;++x) {
            const auto p=image_.pixels[static_cast<std::size_t>(y)*image_.width+x];
            const Vec3 color{p.x,p.y,p.z};
            if (!finite(color)||std::min({p.x,p.y,p.z})<0) throw std::runtime_error("Non-finite or negative HDR environment pixel");
            const double light=luminance(color)*intensity_;
            integral_+=light*omega;
            // A tiny floor keeps filtered radiance supported even beside a black texel.
            weightSum_+=(light+1e-9)*omega;
            cdf_.push_back(weightSum_);
        }
    }
}
// World Y is up; v=0 is the north pole. Subtract yaw on lookup so it is the
// inverse of the rotation applied when an environment direction is sampled.
Vec2 Environment::uv(Vec3 d) const {
    float u=(std::atan2(d.z,d.x)-rotation_)/(2*pi)+0.5f;
    u-=std::floor(u);
    return {u,std::acos(std::clamp(d.y,-1.0f,1.0f))/pi};
}
// Bilinear HDR lookup wraps horizontally across the longitude seam and clamps
// at the poles. HDR values are already linear, so no sRGB conversion occurs here.
Vec3 Environment::radiance(Vec3 d) const {
    if (image_.pixels.empty()) return constant_;
    const Vec2 coord=uv(d);
    const float px=coord.x*static_cast<float>(image_.width)-0.5f, py=coord.y*static_cast<float>(image_.height)-0.5f;
    const int x=static_cast<int>(std::floor(px)), y=static_cast<int>(std::floor(py));
    const float tx=px-static_cast<float>(x), ty=py-static_cast<float>(y);
    auto texel=[&](int ix,int iy) {
        ix=(ix%image_.width+image_.width)%image_.width;
        iy=std::clamp(iy,0,image_.height-1);
        auto p=image_.pixels[static_cast<std::size_t>(iy)*image_.width+ix]; return Vec3{p.x,p.y,p.z};
    };
    return mix(mix(texel(x,y),texel(x+1,y),tx),mix(texel(x,y+1),texel(x+1,y+1),tx),ty)*intensity_;
}
// Pick a texel by importance, then a direction uniformly within its spherical
// area. Interpolating cos(theta), rather than theta, makes that second step uniform.
Vec3 Environment::sample(Rng& rng, float& density) const {
    if (image_.pixels.empty()) {
        const float y=1-2*rng.uniform(), phi=2*pi*rng.uniform(), r=std::sqrt(std::max(0.0f,1-y*y));
        density=1/(4*pi); return {r*std::cos(phi),y,r*std::sin(phi)};
    }
    // Invert the cumulative distribution with one random number and binary search.
    const double pick=rng.uniform()*weightSum_;
    auto it=std::upper_bound(cdf_.begin(),cdf_.end(),pick);
    const std::size_t index=std::min(static_cast<std::size_t>(it-cdf_.begin()),cdf_.size()-1);
    const int x=static_cast<int>(index%image_.width), y=static_cast<int>(index/image_.width);
    const float u=(static_cast<float>(x)+rng.uniform())/static_cast<float>(image_.width);
    const float c0=std::cos(pi*static_cast<float>(y)/static_cast<float>(image_.height));
    const float c1=std::cos(pi*static_cast<float>(y+1)/static_cast<float>(image_.height));
    const float cosTheta=c0+(c1-c0)*rng.uniform(), sinTheta=std::sqrt(std::max(0.0f,1-cosTheta*cosTheta));
    const float phi=(u-0.5f)*2*pi+rotation_;
    density=static_cast<float>((cdf_[index]-(index?cdf_[index-1]:0))/(weightSum_*solidAngle(y)));
    return {sinTheta*std::cos(phi),cosTheta,sinTheta*std::sin(phi)};
}
// Evaluate the same piecewise-constant directional density for a BSDF-selected
// direction. MIS needs this even when the environment did not generate the ray.
float Environment::pdf(Vec3 d) const {
    if (image_.pixels.empty()) return 1/(4*pi);
    const Vec2 coord=uv(d);
    const int x=std::min(static_cast<int>(coord.x*static_cast<float>(image_.width)),image_.width-1);
    const int y=std::min(static_cast<int>(coord.y*static_cast<float>(image_.height)),image_.height-1);
    const auto index=static_cast<std::size_t>(y)*image_.width+x;
    return static_cast<float>((cdf_[index]-(index?cdf_[index-1]:0))/(weightSum_*solidAngle(y)));
}
// Reject invalid dimensions/counts before allocation. Widen the image product
// first so a large requested image cannot wrap a 32-bit integer into a small one.
void RenderOptions::validate() const {
    if (width<1||height<1||width>16384||height>16384) throw std::runtime_error("Width and height must be in [1, 16384]");
    const uint64_t pixels=static_cast<uint64_t>(width)*static_cast<uint64_t>(height);
    if (pixels>67108864 || pixels>std::numeric_limits<std::size_t>::max()/sizeof(Vec3)) throw std::runtime_error("Image exceeds the 64-megapixel allocation limit");
    if (samples<1||samples>1048576) throw std::runtime_error("Samples must be in [1, 1048576]");
    if (bounces<1||bounces>64) throw std::runtime_error("Bounces must be in [1, 64]");
    if (threads<0||threads>256) throw std::runtime_error("Threads must be 0 (automatic) or in [1, 256]");
}
namespace {
// Only boundary inconsistencies trigger precision replay. Other failures remain
// ordinary errors, and a boundary inconsistency surviving replay still fails.
struct BoundaryError : std::runtime_error { using std::runtime_error::runtime_error; };
constexpr uint32_t noTriangle=std::numeric_limits<uint32_t>::max();
// One next-event-estimation proposal. Finite lights carry an endpoint for a
// bounded visibility ray; sun/sky directions extend to infinity. pdf includes
// light-selection probability, with discrete probabilities for delta lights.
struct LightSample {
    Vec3 direction, radiance, target, targetNormal;
    float distance=infinity, pdf=0;
    bool delta=false;
};
// A shared sampler for punctual lights, emissive triangles, and the environment.
// Reverse probability tables let later BSDF hits use the same MIS densities.
class Lights {
    enum class Type { Punctual, Triangle, Environment };
    struct Entry { Type type; uint32_t index; double cdf; float probability=0; };
    std::vector<Entry> entries_;
    std::vector<float> triangleProbability_;
    float environmentProbability_=0;
    const Scene& scene_;
    const Environment& environment_;
public:
    // Approximate emitted power chooses a useful mixture, not an illumination
    // multiplier. Scene radius gives infinite lights a comparable flux scale.
    Lights(const Scene& scene, const Environment& environment) : triangleProbability_(scene.triangles.size()),scene_(scene),environment_(environment) {
        Bounds box;
        for (const auto& tri:scene.triangles) box.expand(tri.bounds());
        const double radius2=scene.triangles.empty()?1:std::max(1.0,static_cast<double>(lengthSquared(box.hi-box.lo))*0.25);
        double total=0;
        // Non-emitting entries are omitted, avoiding zero-probability CDF intervals.
        auto add=[&](Type type, uint32_t index, double weight) {
            if (weight>0) { total+=weight; entries_.push_back({type,index,total,0}); }
        };
        for (uint32_t i=0;i<scene.lights.size();++i) {
            const auto& l=scene.lights[i];
            add(Type::Punctual,i,static_cast<double>(luminance(l.color))*l.intensity*(l.type==LightType::Directional?pi*radius2:4*pi));
        }
        for (uint32_t i=0;i<scene.triangles.size();++i) {
            const auto& tri=scene.triangles[i]; const auto& m=scene.materials[tri.material];
            add(Type::Triangle,i,static_cast<double>(tri.area)*pi*luminance(m.emission)*(m.doubleSided?2:1));
        }
        add(Type::Environment,0,environment.integral()*pi*radius2);
        double last=0, cumulative=0;
        for (auto& e:entries_) {
            // A small uniform component prevents tiny local lights from becoming
            // extremely rare, high-weight samples beside a large environment.
            const double probability=0.95*(e.cdf-last)/total+0.05/static_cast<double>(entries_.size());
            last=e.cdf; cumulative+=probability;
            e.probability=static_cast<float>(probability); e.cdf=cumulative;
            if (e.type==Type::Triangle) triangleProbability_[e.index]=e.probability;
            if (e.type==Type::Environment) environmentProbability_=e.probability;
        }
        if (!entries_.empty()) entries_.back().cdf=1;
    }
    // Joint probability: select the sky, then select this direction within it.
    float environmentPdf(Vec3 direction) const { return environmentProbability_*environment_.pdf(direction); }
    // Convert uniform surface-area density into solid-angle density at 'from':
    // p_omega = p_area * distance^2 / abs(lightNormal dot direction).
    float trianglePdf(uint32_t index, Vec3 from, Vec3 point) const {
        const auto& t=scene_.triangles[index];
        const Vec3 dir=point-from;
        const float len2=lengthSquared(dir), cosine=std::abs(dot(t.geometricNormal,normalize(dir)));
        return cosine>1e-8f?triangleProbability_[index]*len2/(t.area*cosine):0;
    }
    // First choose a light, then generate the appropriate point/direction sample.
    LightSample sample(Vec3 from, Rng& rng) const {
        LightSample out;
        if (entries_.empty()) return out;
        const double u=rng.uniform();
        auto it=std::lower_bound(entries_.begin(),entries_.end(),u,[](const Entry& e,double v){return e.cdf<v;});
        const auto& e=it==entries_.end()?entries_.back():*it;
        if (e.type==Type::Environment) {
            out.direction=environment_.sample(rng,out.pdf);
            out.pdf*=e.probability; out.radiance=environment_.radiance(out.direction); return out;
        }
        if (e.type==Type::Punctual) {
            // Punctual lights are delta distributions. Directional lights do not
            // fall off with distance; point/spot lights use inverse-square falloff.
            const auto& l=scene_.lights[e.index];
            out.delta=true; out.pdf=e.probability;
            float attenuation=1;
            if (l.type==LightType::Directional) out.direction=-l.direction;
            else {
                const Vec3 delta=l.position-from;
                const float r2=lengthSquared(delta);
                if (r2<1e-12f) return {};
                out.distance=std::sqrt(r2); out.direction=delta/out.distance; out.target=l.position;
                attenuation=1/r2;
                if (std::isfinite(l.range)) {
                    // glTF range smoothly suppresses illumination near its cutoff.
                    const float q=out.distance/l.range;
                    attenuation*=std::max(0.0f,1-q*q*q*q);
                }
                if (l.type==LightType::Spot) {
                    // Cone limits are stored as angles; their cosine interval
                    // supplies a smooth angular transition between inner and outer cones.
                    const float cosAngle=dot(l.direction,-out.direction), outer=std::cos(l.outerCone), inner=std::cos(l.innerCone);
                    const float falloff=std::clamp((cosAngle-outer)/std::max(1e-7f,inner-outer),0.0f,1.0f);
                    attenuation*=falloff*falloff;
                }
            }
            out.radiance=l.color*(l.intensity*attenuation); return out;
        }
        const auto& t=scene_.triangles[e.index];
        // The square-root warp gives uniform barycentric samples over triangle area.
        const float r=std::sqrt(rng.uniform()), u1=1-r, u2=r*rng.uniform(), u0=1-u1-u2;
        out.target=t.v[0].position*u0+t.v[1].position*u1+t.v[2].position*u2;
        out.targetNormal=t.geometricNormal;
        const Vec3 delta=out.target-from;
        out.distance=length(delta);
        if (out.distance<1e-7f) return {};
        out.direction=delta/out.distance;
        const Hit hit{out.distance,u1,u2,e.index}; const Ray ray{from,out.direction};
        // Emissive cutouts and one-sided materials obey the same visibility
        // rules as geometry hit by ordinary camera and continuation rays.
        if (!scene_.accepts(t,ray,hit)) return {};
        const Surface s=surfaceAt(scene_,ray,hit);
        if (!s.frontFace&&!s.doubleSided) return {};
        out.radiance=s.emission;
        out.pdf=trianglePdf(e.index,from,out.target);
        return out;
    }
};
// Trace a single camera sample. 'throughput' is the accumulated path weight;
// 'radiance' collects emitted/direct light along the path. Only one homogeneous,
// non-scattering volume can be active because nested volumes are out of scope.
Vec3 trace(const Scene& scene, const Bvh& bvh, const Environment& environment, const Lights& lights,
           Ray ray, const RenderOptions& options, Rng& rng, bool preciseOrigins=false) {
    Vec3 radiance{}, throughput{1}, lastPosition{}, absorption{};
    float lastPdf=0, etaScale=1;
    bool lastDelta=true;
    int medium=-1;
    uint32_t previousTriangle=noTriangle;
    for (int depth=0;depth<=options.bounces;++depth) {
        Hit hit;
        if (!bvh.intersect(scene,ray,hit,false,previousTriangle)) {
            // A miss sees the environment. Weight it against the light sampler
            // unless the previous event was the camera or a specular delta event.
            if (medium>=0) throughput*=beerLambert(absorption,infinity);
            const float weight=lastDelta||!options.nextEventEstimation?1:powerHeuristic(lastPdf,lights.environmentPdf(ray.direction));
            radiance+=throughput*environment.radiance(ray.direction)*weight;
            break;
        }
        // Extinction applies to the distance actually traveled before this hit.
        if (medium>=0) throughput*=beerLambert(absorption,static_cast<float>(hit.t));
        const Surface s=surfaceAt(scene,ray,hit);
        if (s.frontFace || s.doubleSided) {
            // Seeing an emitter through BSDF sampling competes with explicitly
            // sampling that same emitter; MIS prevents counting its energy twice.
            const float weight=lastDelta||!options.nextEventEstimation?1:powerHeuristic(lastPdf,lights.trianglePdf(hit.triangle,lastPosition,s.position));
            radiance+=throughput*s.emission*weight;
        }
        // The last permitted segment may see emission but does not scatter again.
        if (depth==options.bounces) break;
        const Vec3 wo=-ray.direction;
        if (options.nextEventEstimation) {
            // Estimate direct lighting by sampling a light independently of the
            // continuation direction. The matching BSDF PDF supplies the MIS weight.
            const auto light=lights.sample(s.position,rng);
            // An infinite shadow ray from an opaque surface inside a closed
            // volume must cross its dielectric boundary, which blocks straight
            // NEE visibility. BSDF paths still reach the sky through refraction.
            const bool enclosedInfinite=medium>=0 && s.thickness<=0 && !std::isfinite(light.distance);
            if (!enclosedInfinite && light.pdf>0 && maxComponent(light.radiance)>0) {
                const auto bsdf=evaluateBsdf(s,wo,light.direction);
                if (maxComponent(bsdf.value)>0) {
                    Ray shadow=preciseOrigins?Ray{s.position,light.direction,0,infinity,s.positionLow}:offsetRay(s.position,s.geometricNormal,light.direction);
                    if (std::isfinite(light.distance)) {
                        // Offset both ends and stop short of the light so its own
                        // emitting triangle is not mistaken for a shadow blocker.
                        Vec3 end=light.target;
                        if (lengthSquared(light.targetNormal)>0) end=offsetRay(end,light.targetNormal,-light.direction).origin;
                        const Vec3 delta=end-shadow.origin;
                        shadow.tmax=length(delta)*(1-2e-6f); shadow.direction=normalize(delta);
                    }
                    Hit blocker;
                    if (!bvh.intersect(scene,shadow,blocker,true,preciseOrigins?hit.triangle:noTriangle)) {
                        const float weight=light.delta?1:powerHeuristic(light.pdf,bsdf.pdf);
                        Vec3 shadowAbsorption=absorption;
                        bool shadowInMedium=medium>=0;
                        // A transmitted shadow direction lies on the OTHER side
                        // of this interface. Evaluate its medium without changing
                        // the continuing path's medium state.
                        if (s.thickness>0 && dot(wo,s.geometricNormal)*dot(light.direction,s.geometricNormal)<0) {
                            shadowInMedium=s.frontFace;
                            shadowAbsorption=s.frontFace?absorptionCoefficient(s.attenuationColor,s.attenuationDistance):Vec3{};
                        }
                        const Vec3 transmittance=shadowInMedium?beerLambert(shadowAbsorption,light.distance):Vec3{1};
                        radiance+=throughput*bsdf.value*light.radiance*transmittance*(std::abs(dot(s.normal,light.direction))*weight/light.pdf);
                    }
                }
            }
        }
        // Continue the path with one BSDF event, whose returned weight already
        // contains its cosine factor and reciprocal sampling probability.
        const auto sample=sampleBsdf(s,wo,rng);
        if (!sample) break;
        throughput*=sample.weight;
        if (!finite(throughput)) throw std::runtime_error("Non-finite path throughput");
        if (sample.transmitted && s.thickness>0) {
            // Only an actual closed-volume crossing changes medium/IOR state.
            // Opaque reflections and thin-sheet transmission leave it unchanged.
            if (s.frontFace) {
                if (medium>=0) throw BoundaryError("Nested/intersecting refractive volumes are unsupported; separate the closed meshes");
                medium=static_cast<int>(s.material); absorption=absorptionCoefficient(s.attenuationColor,s.attenuationDistance);
            } else {
                if (medium<0) {
                    // Include enough geometric context to reproduce a failing
                    // ray; render() adds its pixel and sample coordinates.
                    std::ostringstream detail;
                    detail.precision(9);
                    detail << "Refractive exit without entry; check outward normals and keep the camera outside closed volumes"
                           << "; bounce " << depth << ", triangle " << hit.triangle << ", material " << s.material
                           << ", t " << hit.t << ", point (" << s.position.x << ',' << s.position.y << ',' << s.position.z
                           << "), origin (" << ray.origin.x << ',' << ray.origin.y << ',' << ray.origin.z
                           << "), direction (" << ray.direction.x << ',' << ray.direction.y << ',' << ray.direction.z << ')';
                    throw BoundaryError(detail.str());
                }
                medium=-1; absorption={};
            }
            // Undo radiance-mode eta scaling only for roulette's survival estimate.
            etaScale*=sample.eta*sample.eta;
        }
        // Remember this event for emitter/sky MIS on the following segment.
        lastPdf=sample.pdf; lastDelta=sample.delta; lastPosition=s.position;
        ray=preciseOrigins?Ray{s.position,sample.direction,0,infinity,s.positionLow}:offsetRay(s.position,s.geometricNormal,sample.direction);
        // Split origins retain low coordinate bits on a precision replay. A ray
        // leaving a flat triangle cannot legitimately re-hit that same triangle
        // before another event, so only the immediately departed primitive is excluded.
        previousTriangle=preciseOrigins?hit.triangle:noTriangle;
        if (options.russianRoulette && depth>=3) {
            // Shorten low-contribution paths stochastically. Dividing survivors
            // by their survival probability preserves the estimator's expectation.
            const float survive=std::clamp(maxComponent(throughput)*etaScale,0.05f,0.95f);
            if (rng.uniform()>=survive) break;
            throughput*=1/survive;
        }
    }
    // Do not silently replace broken transport with black pixels or clamp it away.
    if (!finite(radiance)||std::min({radiance.x,radiance.y,radiance.z})<0) throw std::runtime_error("Non-finite or negative path radiance");
    return radiance;
}
}
// Allocate the linear film, share read-only scene/light data, and schedule tiles.
// Thread scheduling never participates in seed generation or pixel accumulation.
RenderResult render(const Scene& scene, const Bvh& bvh, const Camera& camera, const Environment& environment,
                    const RenderOptions& options, Progress progress) {
    options.validate();
    RenderResult result;
    const std::size_t pixels=static_cast<std::size_t>(options.width)*options.height;
    result.pixels.resize(pixels);
    result.cameraSamples=static_cast<uint64_t>(pixels)*options.samples;
    // Small tiles balance expensive water/foliage pixels against cheap sky pixels.
    constexpr int tileSize=16;
    const int tilesX=(options.width+tileSize-1)/tileSize, tilesY=(options.height+tileSize-1)/tileSize, tiles=tilesX*tilesY;
    // Leave scheduling headroom for the desktop and other applications by default.
    result.workers=std::min(tiles,options.threads?options.threads:static_cast<int>(std::max(1u,std::thread::hardware_concurrency()/2)));
    const Lights lights(scene,environment);
    // Only scheduling, progress, cancellation, and diagnostics are shared writes.
    // Film writes need no lock because a tile is assigned to exactly one worker.
    std::atomic<int> nextTile{0};
    std::atomic<uint64_t> finished{0};
    std::atomic<uint64_t> boundaryRetries{0};
    std::atomic<bool> cancel{false};
    std::mutex stateMutex;
    std::exception_ptr error;
    const auto start=std::chrono::steady_clock::now();
    auto lastProgress=start;
    auto worker=[&] {
        // Retain the current sample coordinates for actionable exception messages.
        int currentX=-1,currentY=-1,currentSample=-1;
        try {
            for (;;) {
                const int tile=nextTile.fetch_add(1,std::memory_order_relaxed);
                if (tile>=tiles || cancel.load(std::memory_order_relaxed)) break;
                const int x0=(tile%tilesX)*tileSize, y0=(tile/tilesX)*tileSize;
                const int x1=std::min(x0+tileSize,options.width), y1=std::min(y0+tileSize,options.height);
                for (int y=y0;y<y1;++y) for (int x=x0;x<x1;++x) {
                    currentX=x; currentY=y;
                    if (cancel.load(std::memory_order_relaxed)) return;
                    const auto pixel=static_cast<std::size_t>(y)*options.width+x;
                    // Sum this pixel in double precision and in a fixed order;
                    // do not merge floating-point partial sums from different threads.
                    double r=0,g=0,b=0;
                    for (int sample=0;sample<options.samples;++sample) {
                        currentSample=sample;
                        if ((sample&255)==0 && cancel.load(std::memory_order_relaxed)) return;
                        // Jitter within the pixel for antialiasing. This key gives
                        // every (seed, pixel, sample) its own reproducible random stream.
                        Rng rng(sampleSeed(options.seed,pixel,static_cast<uint64_t>(sample)));
                        const float dx=rng.uniform(), dy=rng.uniform();
                        const Ray ray=camera.generate(static_cast<float>(x)+dx,static_cast<float>(y)+dy,options.width,options.height);
                        // Save the post-camera RNG state so a precision replay
                        // uses the same camera ray and random sample, not a replacement.
                        const Rng initial=rng;
                        Vec3 light;
                        try { light=trace(scene,bvh,environment,lights,ray,options,rng); }
                        catch (const BoundaryError&) {
                            // An offset can jump an adjacent interface where opaque
                            // geometry meets water. Replay the same sample from the
                            // actual surfaces, excluding only the departed triangle.
                            // Invalid cameras/volumes still fail on this second pass.
                            boundaryRetries.fetch_add(1,std::memory_order_relaxed);
                            rng=initial;
                            light=trace(scene,bvh,environment,lights,ray,options,rng,true);
                        }
                        r+=light.x; g+=light.y; b+=light.z;
                    }
                    result.pixels[pixel]={static_cast<float>(r/options.samples),static_cast<float>(g/options.samples),static_cast<float>(b/options.samples)};
                }
                const uint64_t tileSamples=static_cast<uint64_t>((x1-x0)*(y1-y0))*options.samples;
                finished.fetch_add(tileSamples,std::memory_order_relaxed);
                if (progress) {
                    // Serialize callbacks and throttle output; rendering itself
                    // stays outside this lock.
                    std::lock_guard lock(stateMutex);
                    const auto now=std::chrono::steady_clock::now();
                    if (now-lastProgress>=std::chrono::seconds(2)) {
                        lastProgress=now; progress(finished.load(),result.cameraSamples,std::chrono::duration<double>(now-start).count());
                    }
                }
            }
        } catch (...) {
            // Stop peers promptly and retain the first failure. Exceptions are
            // rethrown on the caller thread only after every worker has joined.
            cancel.store(true);
            std::lock_guard lock(stateMutex);
            if (!error) {
                try { throw; }
                catch (const std::exception& cause) {
                    error=std::make_exception_ptr(std::runtime_error("Pixel ("+std::to_string(currentX)+","+std::to_string(currentY)
                        +"), sample "+std::to_string(currentSample)+": "+cause.what()));
                }
                catch (...) { error=std::current_exception(); }
            }
        }
    };
    std::vector<std::jthread> workers;
    workers.reserve(static_cast<std::size_t>(result.workers));
    try { for (int i=0;i<result.workers;++i) workers.emplace_back(worker); }
    // jthread destruction joins running workers even if thread creation fails.
    catch (...) { cancel.store(true); workers.clear(); throw; }
    workers.clear();
    // No film writes remain after this point, so the result can be safely encoded.
    if (error) std::rethrow_exception(error);
    result.boundaryRetries=boundaryRetries.load();
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (progress) progress(result.cameraSamples,result.cameraSamples,result.seconds);
    return result;
}
}
