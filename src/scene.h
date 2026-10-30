#pragma once
#include "math.h"
#include <filesystem>
#include <string>
#include <vector>

namespace cr {
// A ray evaluates to origin + originLow + t*direction on the inclusive interval.
// Rendering produces unit directions, so t then measures world-space distance.
struct Ray {
    Vec3 origin, direction; float tmin=0, tmax=infinity;
    // Low part of the reconstructed origin, used for near-interface retracing.
    Vec3 originLow{};
};
// Axis-aligned bounds start empty; expansion accumulates geometry or child boxes.
// Surface area feeds SAH construction and intersection supplies traversal ordering.
struct Bounds {
    Vec3 lo{infinity}, hi{-infinity};
    void expand(Vec3 p) { lo=min(lo,p); hi=max(hi,p); }
    void expand(const Bounds& b) { lo=min(lo,b.lo); hi=max(hi,b.hi); }
    float area() const { Vec3 d=max(hi-lo,Vec3{}); return 2*(d.x*d.y+d.y*d.z+d.z*d.x); }
    bool intersect(const Ray& ray, float& nearDistance) const;
};
// World-space vertex data after node transforms. tangent.w carries TBN handedness;
// two UV sets cover the supported glTF texture-coordinate selections.
struct Vertex { Vec3 position, normal; Vec4 tangent{0,0,0,1}; Vec2 uv[2]{}; };
// Flattened triangles reference renderer-owned materials. finalize() computes
// their outward geometric normals and areas before rendering/BVH construction.
struct Triangle {
    Vertex v[3]; uint32_t material=0;
    Vec3 geometricNormal; float area=0;
    Bounds bounds() const { Bounds b; for (const auto& x:v) b.expand(x.position); return b; }
    Vec3 centroid() const { return (v[0].position+v[1].position+v[2].position)/3; }
};
// u/v weight vertices 1/2, with vertex 0 weighted by 1-u-v. Double precision
// preserves ordering and reconstruction when interfaces are closer than float t.
struct Hit { double t=infinity, u=0, v=0; uint32_t triangle=0; };
// Primitive intersection is geometric/two-sided; Scene::accepts applies materials.
bool intersectTriangle(const Triangle&, const Ray&, Hit&);
// Normal-based, scale-aware bias keeps ordinary secondary/shadow rays off their
// source surface. Very close interfaces additionally use precise-origin replay.
Ray offsetRay(Vec3 position, Vec3 geometricNormal, Vec3 direction, float maxDistance=infinity);

enum class Wrap { Repeat, Clamp, Mirror };
// Decoded LDR channel values remain sRGB-encoded where applicable; HDR is linear.
// Conversion is decided by each material binding so data maps are never gamma-corrected.
struct Image { int width=0, height=0; std::vector<Vec4> pixels; };
struct Texture { int image=-1; Wrap wrapS=Wrap::Repeat, wrapT=Wrap::Repeat; bool linear=true; };
// A binding adds UV selection/transform to a shared image+sampler. -1 means no map.
struct TextureSlot {
    int texture=-1, texcoord=0;
    Vec2 offset{}, scale{1,1}; float rotation=0;
};
enum class AlphaMode { Opaque, Mask };
// Supported metallic/roughness material factors and texture bindings. Color and
// emission factors are linear; normal, roughness, metal, transmission, and alpha
// channels are data. Unsupported glTF material models are rejected by the adapter.
struct Material {
    std::string name;
    Vec4 baseColor{1,1,1,1};
    float metallic=1, roughness=1;
    Vec3 emission{};
    float transmission=0, ior=1.5f;
    // thickness > 0 denotes a closed volume; actual world-space path length controls absorption.
    float thickness=0, attenuationDistance=infinity;
    Vec3 attenuationColor{1};
    bool doubleSided=false;
    AlphaMode alphaMode=AlphaMode::Opaque; float alphaCutoff=0.5f, normalScale=1;
    TextureSlot baseColorTexture, metallicRoughnessTexture, normalTexture, emissiveTexture, transmissionTexture;
};
// Perspective camera in world coordinates. aspect=0 uses output dimensions;
// near/far clipping describes planes perpendicular to forward, not spheres.
struct Camera {
    std::string name;
    Vec3 position{}, forward{0,0,-1}, right{1,0,0}, up{0,1,0};
    float yfov=pi/4, aspect=0, nearClip=0.01f, farClip=infinity;
    Ray generate(float pixelX, float pixelY, int width, int height) const;
};
enum class LightType { Point, Directional, Spot };
// Punctual lights are delta emitters. Their transformed direction is the direction
// the light shines; the integrator samples the opposite direction toward the light.
struct PunctualLight {
    LightType type=LightType::Point;
    Vec3 position, direction{0,0,-1}, color{1};
    float intensity=1, range=infinity, innerCone=0, outerCone=pi/4;
};
// All imported data is owned here after parser storage is released. No glTF or
// image-library pointers escape into the renderer or its worker threads.
struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<Image> images;
    std::vector<Texture> textures;
    std::vector<Camera> cameras;
    std::vector<PunctualLight> lights;
    // Used by the CLI to protect every loaded file from accidental output overwrite.
    std::vector<std::filesystem::path> resources;
    std::vector<std::string> warnings;
    // Common texture sampling and acceptance rules serve every ray type.
    Vec4 sample(const TextureSlot&, Vec2 uv0, Vec2 uv1, bool srgb) const;
    bool accepts(const Triangle&, const Ray&, const Hit&) const;
    // Validate inputs, remove degenerate triangles, and prepare usable frames.
    void finalize();
};
// Evaluated material/frame at one hit, ready for the BSDF. Geometric and shading
// normals remain outward; frontFace records the ray's side instead of flipping them.
struct Surface {
    Vec3 position, geometricNormal, normal, tangent, bitangent;
    // Residual of the double-precision position after its high part is rounded.
    Vec3 positionLow{};
    Vec3 baseColor, emission, attenuationColor;
    float metallic=0, roughness=1, transmission=0, ior=1.5f, thickness=0, attenuationDistance=infinity;
    uint32_t material=0;
    bool frontFace=true, doubleSided=false;
};
// Interpolate vertex attributes, apply material textures, and construct the TBN frame.
Surface surfaceAt(const Scene&, const Ray&, const Hit&);

// Binned surface-area-heuristic accelerator over static world-space triangles.
// The same tree supports nearest hits and early-out shadow visibility queries.
class Bvh {
public:
    void build(const Scene&);
    // ignoreTriangle is used only for the immediately departed primitive during
    // precision replay. Alpha masks and sidedness still apply to every other hit.
    bool intersect(const Scene&, const Ray&, Hit&, bool any=false,
                   uint32_t ignoreTriangle=std::numeric_limits<uint32_t>::max()) const;
    std::size_t nodeCount() const { return nodes_.size(); }
    std::size_t bytes() const;
private:
    // count>0 denotes a range in indices_; count=0 denotes two adjacent child nodes.
    struct Node { Bounds box; uint32_t first=0, count=0; };
    std::vector<Node> nodes_;
    std::vector<uint32_t> indices_;
    void subdivide(const Scene&, uint32_t node, uint32_t begin, uint32_t end, int depth);
};

namespace io {
// Adapter boundary: local glTF loading and image encoding/decoding. Implementations
// alone include third-party headers; callers operate entirely on the types above.
Scene loadGltf(const std::filesystem::path& path);
Image loadImage(const std::filesystem::path& path, bool hdr=false);
Image decodeImage(const unsigned char* bytes, std::size_t size, const std::string& label, bool hdr=false);
void writePng(const std::filesystem::path&, int width, int height, const std::vector<Vec3>& linear, float exposure);
// HDR is written from the original unexposed linear film, not from the display PNG.
void writeHdr(const std::filesystem::path&, int width, int height, const std::vector<Vec3>& linear);
}
}
