#pragma once
#include "bsdf.h"
#include <functional>

namespace cr {
// Infinite lighting, represented either by a constant radiance or a linear HDR
// latitude/longitude image. Lookup, sampling, and PDFs share the same Y-up mapping.
class Environment {
public:
    // The default is black, not an implicit scene-specific sky or ambient term.
    explicit Environment(Vec3 constant=Vec3{});
    Environment(Image image, float rotationDegrees, float intensity);
    Vec3 radiance(Vec3 direction) const;
    Vec3 sample(Rng&, float& pdf) const;
    float pdf(Vec3 direction) const;
    // Integrated luminance weights light selection; bytes() reports owned storage.
    double integral() const { return integral_; }
    std::size_t bytes() const { return image_.pixels.size()*sizeof(Vec4)+cdf_.size()*sizeof(double); }
private:
    Image image_;
    Vec3 constant_{};
    float rotation_=0, intensity_=1;
    double integral_=0, weightSum_=0;
    // Cumulative texel importance: luminance multiplied by spherical solid angle.
    std::vector<double> cdf_;
    Vec2 uv(Vec3 direction) const;
    double solidAngle(int row) const;
};
// All runtime quality/scheduling controls. The two estimator switches are also
// useful in tests that compare MIS against pure BSDF sampling and disable roulette.
struct RenderOptions {
    // Automatic (threads=0) uses half the logical CPUs, with a minimum of one.
    // Explicit positive counts remain available for deliberate offline benchmarks.
    int width=960, height=540, samples=32, bounces=8, threads=0;
    uint64_t seed=42;
    bool nextEventEstimation=true, russianRoulette=true;
    // Validate before allocating a film or starting any worker threads.
    void validate() const;
};
// Film values remain linear and unexposed. Image encoding/tonemapping lives in io.
struct RenderResult {
    std::vector<Vec3> pixels;
    double seconds=0;
    int workers=0;
    uint64_t cameraSamples=0;
    // Counts full-sample precision replays, not skipped or clamped samples.
    uint64_t boundaryRetries=0;
};
// Reports completed camera samples and elapsed worker time, not individual rays.
using Progress=std::function<void(uint64_t complete, uint64_t total, double seconds)>;
// Scene/BVH/camera/environment are immutable and shared across workers. Each pixel
// is owned by one worker and summed in a fixed sample order for deterministic output.
RenderResult render(const Scene&, const Bvh&, const Camera&, const Environment&, const RenderOptions&, Progress progress={});
}
