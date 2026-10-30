#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cr {
// Small renderer-owned value types keep geometry/color math independent of an
// asset parser or a platform-specific graphics API. Vec3 also represents RGB.
constexpr float pi = 3.14159265358979323846f;
constexpr float infinity = std::numeric_limits<float>::infinity();
struct Vec2 { float x = 0, y = 0; };
struct Vec3 {
    float x = 0, y = 0, z = 0;
    constexpr Vec3() = default;
    constexpr explicit Vec3(float s) : x(s), y(s), z(s) {}
    constexpr Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};
struct Vec4 { float x = 0, y = 0, z = 0, w = 1; };
// Component arithmetic supports both positions/vectors and RGB throughput.
// A vector-vector product multiplies components; it is not a dot product.
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x+b.x,a.y+b.y}; }
inline Vec2 operator*(Vec2 a, float b) { return {a.x*b,a.y*b}; }
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x,-a.y,-a.z}; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x*b.x,a.y*b.y,a.z*b.z}; }
inline Vec3 operator/(Vec3 a, Vec3 b) { return {a.x/b.x,a.y/b.y,a.z/b.z}; }
inline Vec3 operator*(Vec3 a, float b) { return {a.x*b,a.y*b,a.z*b}; }
inline Vec3 operator*(float a, Vec3 b) { return b*a; }
inline Vec3 operator/(Vec3 a, float b) { return a*(1.0f/b); }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a=a+b; return a; }
inline Vec3& operator*=(Vec3& a, Vec3 b) { a=a*b; return a; }
inline Vec3& operator*=(Vec3& a, float b) { a=a*b; return a; }
// Geometric products: dot measures projection/angle, cross builds an oriented
// perpendicular vector, and length/normalize convert a direction to unit scale.
inline float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline float lengthSquared(Vec3 a) { return dot(a,a); }
inline float length(Vec3 a) { return std::sqrt(lengthSquared(a)); }
inline Vec3 normalize(Vec3 a) { const float l=length(a); return l>0 ? a/l : Vec3{}; }
// Component-wise bounds and validity helpers are shared by geometry and shading.
inline bool finite(Vec3 a) { return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z); }
inline Vec3 min(Vec3 a, Vec3 b) { return {std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)}; }
inline Vec3 max(Vec3 a, Vec3 b) { return {std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)}; }
inline Vec3 clamp(Vec3 a, float lo, float hi) { return {std::clamp(a.x,lo,hi),std::clamp(a.y,lo,hi),std::clamp(a.z,lo,hi)}; }
inline float maxComponent(Vec3 a) { return std::max({a.x,a.y,a.z}); }
// Linear RGB luminance weights guide importance sampling, not color conversion.
inline float luminance(Vec3 a) { return dot(a,{0.2126f,0.7152f,0.0722f}); }
inline Vec3 mix(Vec3 a, Vec3 b, float t) { return a*(1-t)+b*t; }
// Beer attenuation applies an exponential independently to each color channel.
inline Vec3 exp(Vec3 a) { return {std::exp(a.x),std::exp(a.y),std::exp(a.z)}; }
inline Vec3 reflect(Vec3 incident, Vec3 n) { return incident-2*dot(incident,n)*n; }
// Color textures/display PNGs use sRGB; all light transport is linear. The small
// linear segment near zero is part of sRGB, not a generic gamma approximation.
inline float srgbToLinear(float v) { return v<=0.04045f ? v/12.92f : std::pow((v+0.055f)/1.055f,2.4f); }
inline float linearToSrgb(float v) { return v<=0.0031308f ? 12.92f*v : 1.055f*std::pow(v,1/2.4f)-0.055f; }

struct Mat4 {
    // Column-major, as in glTF. Identity is the default.
    std::array<float,16> m{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    // Points include translation (homogeneous w=1); directions do not (w=0).
    Vec3 point(Vec3 p) const {
        return {m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12], m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13], m[2]*p.x+m[6]*p.y+m[10]*p.z+m[14]};
    }
    Vec3 vector(Vec3 p) const {
        return {m[0]*p.x+m[4]*p.y+m[8]*p.z,m[1]*p.x+m[5]*p.y+m[9]*p.z,m[2]*p.x+m[6]*p.y+m[10]*p.z};
    }
    // The determinant sign detects a mirrored basis and hence reversed handedness.
    float determinant() const { return dot(vector({1,0,0}),cross(vector({0,1,0}),vector({0,0,1}))); }
    // Normals require inverse-transpose transformation under nonuniform scale or
    // shear, rather than the ordinary vector transform used for tangent directions.
    Vec3 normal(Vec3 n) const {
        const double a=m[0], b=m[4], c=m[8], d=m[1], e=m[5], f=m[9], g=m[2], h=m[6], i=m[10];
        const double det=a*(e*i-f*h)+b*(f*g-d*i)+c*(d*h-e*g);
        if (!std::isfinite(det)||det==0) throw std::runtime_error("Singular node transform");
        // Only direction is needed: normalize cofactors in double instead of
        // dividing by a scale-dependent determinant that can under/overflow.
        const double x=(e*i-f*h)*n.x+(f*g-d*i)*n.y+(d*h-e*g)*n.z;
        const double y=(c*h-b*i)*n.x+(a*i-c*g)*n.y+(b*g-a*h)*n.z;
        const double z=(b*f-c*e)*n.x+(c*d-a*f)*n.y+(a*e-b*d)*n.z;
        const double magnitude=std::hypot(x,y,z);
        if (magnitude==0) return {};
        const double sign=det<0?-1.0:1.0;
        return {static_cast<float>(sign*x/magnitude),static_cast<float>(sign*y/magnitude),static_cast<float>(sign*z/magnitude)};
    }
};
// Orthonormal shading basis: local +Z is the supplied normal. Choosing a helper
// axis that is not parallel to that normal avoids a degenerate cross product.
struct Frame {
    Vec3 x,y,z;
    explicit Frame(Vec3 normal) : z(normal) {
        x=normalize(cross(std::abs(z.z)<0.999f?Vec3{0,0,1}:Vec3{0,1,0},z)); y=cross(z,x);
    }
    // Dot with basis axes to enter local coordinates; combine axes to leave them.
    Vec3 local(Vec3 v) const { return {dot(v,x),dot(v,y),dot(v,z)}; }
    Vec3 world(Vec3 v) const { return x*v.x+y*v.y+z*v.z; }
};
// SplitMix-style integer generator. Its entire state is explicit, so a sample
// can be replayed exactly and never depends on a worker's execution schedule.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint32_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z=state; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL;
        return static_cast<uint32_t>((z^(z>>31))>>32);
    }
    // Use 24 random bits to produce representable floats in [0,1), never 1.
    float uniform() { return static_cast<float>(next()>>8)*(1.0f/16777216.0f); }
};
// Hash the user seed, pixel index, and sample index into independent path streams.
inline uint64_t sampleSeed(uint64_t seed, uint64_t pixel, uint64_t sample) {
    Rng hash(seed ^ (pixel*0xd2b74407b1ce6e93ULL) ^ (sample*0xca5a826395121157ULL));
    const uint64_t hi=hash.next(); return (hi<<32)|hash.next();
}
// MIS power heuristic for two competing probability densities. Squaring in
// double avoids overflow from sharp glossy PDFs while favoring the better sampler.
inline float powerHeuristic(float a, float b) {
    if (a<=0) return 0;
    const double aa=static_cast<double>(a)*a, bb=static_cast<double>(b)*b;
    return static_cast<float>(aa/(aa+bb));
}
}
