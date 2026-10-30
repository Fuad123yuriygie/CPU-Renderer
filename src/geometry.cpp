// Geometry, texture/frame evaluation, validation, and the static BVH. Storage is
// compact float data; sensitive geometric arithmetic uses double intermediates.
#include "scene.h"

#include <array>
#include <numeric>
#include <utility>

namespace cr {
namespace {
// SAH binning bounds construction cost, while the depth cap bounds traversal stack use.
constexpr int binCount = 16;
constexpr int maxBvhDepth = 48;
constexpr uint32_t leafSize = 4;

// Private high-precision arithmetic does not enlarge every stored mesh vertex.
struct DVec3 {
    double x, y, z;
    explicit DVec3(Vec3 v) : x(v.x), y(v.y), z(v.z) {}
    DVec3(double a, double b, double c) : x(a), y(b), z(c) {}
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
DVec3 operator+(DVec3 a, DVec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
DVec3 operator-(DVec3 a, DVec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
DVec3 operator*(DVec3 a, double b) { return {a.x*b, a.y*b, a.z*b}; }
double preciseDot(DVec3 a, DVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
DVec3 preciseCross(DVec3 a, DVec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
// Normalize safely, returning an explicit fallback for a zero/invalid direction.
Vec3 unit(DVec3 v, Vec3 fallback = {}) {
    const double squared = preciseDot(v, v);
    if (!(squared > 0) || !std::isfinite(squared)) return fallback;
    const double inverse = 1/std::sqrt(squared);
    return {static_cast<float>(v.x*inverse), static_cast<float>(v.y*inverse), static_cast<float>(v.z*inverse)};
}
Vec3 unit(Vec3 v, Vec3 fallback = {}) { return unit(DVec3(v), fallback); }
// Gram-Schmidt removes the component parallel to n. Detect near-parallel input
// before normalization magnifies numerical noise into an arbitrary tangent.
Vec3 orthogonalized(DVec3 v, Vec3 n, Vec3 fallback = {}) {
    const double normalSquared = preciseDot(DVec3(n),DVec3(n));
    if (!(normalSquared > 0)) return fallback;
    const DVec3 projected = v-DVec3(n)*(preciseDot(v,DVec3(n))/normalSquared);
    constexpr double tolerance = 4*std::numeric_limits<float>::epsilon();
    if (preciseDot(projected,projected) <= preciseDot(v,v)*tolerance*tolerance) return fallback;
    return unit(projected,fallback);
}
// Compact validation helpers cover vectors, colors, UVs, and complete ray intervals.
bool nonzero(Vec3 v) { return v.x != 0 || v.y != 0 || v.z != 0; }
bool finite4(Vec4 v) { return finite({v.x,v.y,v.z}) && std::isfinite(v.w); }
bool finite2(Vec2 v) { return std::isfinite(v.x) && std::isfinite(v.y); }
bool validRay(const Ray& ray) {
    return finite(ray.origin) && finite(ray.originLow) && finite(ray.direction) && nonzero(ray.direction) &&
        !std::isnan(ray.tmin) && !std::isnan(ray.tmax) && ray.tmin <= ray.tmax;
}
// Double sums/areas avoid overflow or cancellation while evaluating SAH candidates.
Vec3 centroid(const Triangle& t) {
    const DVec3 c = (DVec3(t.v[0].position)+DVec3(t.v[1].position)+DVec3(t.v[2].position))*(1.0/3.0);
    return {static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z)};
}
double boundsArea(const Bounds& b) {
    if (b.lo.x > b.hi.x || b.lo.y > b.hi.y || b.lo.z > b.hi.z) return 0;
    const DVec3 d = DVec3(b.hi)-DVec3(b.lo);
    return 2*(d.x*d.y+d.y*d.z+d.z*d.x);
}
// Barycentric UV interpolation follows the imported orientation; there is no
// additional image/UV flip after the authoring tool has exported glTF coordinates.
Vec2 interpolateUv(const Triangle& triangle, const Hit& hit, int set) {
    const double w = 1.0-static_cast<double>(hit.u)-hit.v;
    return {
        static_cast<float>(w*triangle.v[0].uv[set].x+static_cast<double>(hit.u)*triangle.v[1].uv[set].x+static_cast<double>(hit.v)*triangle.v[2].uv[set].x),
        static_cast<float>(w*triangle.v[0].uv[set].y+static_cast<double>(hit.u)*triangle.v[1].uv[set].y+static_cast<double>(hit.v)*triangle.v[2].uv[set].y)
    };
}
// Reduce coordinates before converting them to integer texel indices. This also
// makes very large finite UVs and negative repeat/mirror coordinates safe.
double wrappedCoordinate(double value, Wrap wrap) {
    if (wrap == Wrap::Clamp) return std::clamp(value, 0.0, 1.0);
    if (wrap == Wrap::Repeat) return value-std::floor(value);
    double result = std::fmod(value, 2.0);
    if (result < 0) result += 2;
    return result <= 1 ? result : 2-result;
}
// Bilinear neighbors can cross image edges even after UV reduction; wrap/clamp
// those integer neighbors separately to preserve seams and mirrored boundaries.
int wrappedIndex(int64_t index, int size, Wrap wrap) {
    const int64_t n = size;
    if (wrap == Wrap::Clamp) return static_cast<int>(std::clamp(index, int64_t{0}, n-1));
    const int64_t period = wrap == Wrap::Mirror ? 2*n : n;
    index %= period;
    if (index < 0) index += period;
    if (wrap == Wrap::Mirror && index >= n) index = period-1-index;
    return static_cast<int>(index);
}
// Four-channel interpolation keeps alpha linear alongside the sampled RGB data.
Vec4 interpolate(Vec4 a, Vec4 b, float t) {
    return {a.x*(1-t)+b.x*t, a.y*(1-t)+b.y*t, a.z*(1-t)+b.z*t, a.w*(1-t)+b.w*t};
}
// Solve the triangle's UV-to-position differential for a missing tangent frame.
// A singular UV triangle cannot define these directions and needs a fallback.
bool uvDirections(const Triangle& triangle, int set, DVec3& tangent, DVec3& bitangent) {
    const double du1 = static_cast<double>(triangle.v[1].uv[set].x)-triangle.v[0].uv[set].x;
    const double dv1 = static_cast<double>(triangle.v[1].uv[set].y)-triangle.v[0].uv[set].y;
    const double du2 = static_cast<double>(triangle.v[2].uv[set].x)-triangle.v[0].uv[set].x;
    const double dv2 = static_cast<double>(triangle.v[2].uv[set].y)-triangle.v[0].uv[set].y;
    const double determinant = du1*dv2-dv1*du2;
    if (determinant == 0) return false;
    const DVec3 e1 = DVec3(triangle.v[1].position)-DVec3(triangle.v[0].position);
    const DVec3 e2 = DVec3(triangle.v[2].position)-DVec3(triangle.v[0].position);
    // Only the directions are needed; avoid dividing by a very small UV area.
    const double sign = determinant < 0 ? -1 : 1;
    tangent = (e1*dv2-e2*dv1)*sign;
    bitangent = (e2*du1-e1*du2)*sign;
    return nonzero(unit(tangent)) && nonzero(unit(bitangent));
}
}

// Reference slab test: intersect the ray's interval with each coordinate slab.
// Parallel components are handled directly, avoiding reciprocal infinities/NaNs.
bool Bounds::intersect(const Ray& ray, float& nearDistance) const {
    if (!validRay(ray)) return false;
    double nearT = ray.tmin, farT = ray.tmax;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::isnan(lo[axis]) || std::isnan(hi[axis]) || lo[axis] > hi[axis]) return false;
        const double origin = static_cast<double>(ray.origin[axis])+ray.originLow[axis], direction = ray.direction[axis];
        if (direction == 0) {
            if (origin < lo[axis] || origin > hi[axis]) return false;
            continue;
        }
        double a = (static_cast<double>(lo[axis])-origin)/direction;
        double b = (static_cast<double>(hi[axis])-origin)/direction;
        if (a > b) std::swap(a, b);
        // Round slabs outward, including zero-thickness boxes and boundary rays.
        a = std::nextafter(a, -std::numeric_limits<double>::infinity());
        b = std::nextafter(b, std::numeric_limits<double>::infinity());
        nearT = std::max(nearT, a);
        farT = std::min(farT, b);
        if (nearT > farT) return false;
    }
    const float rounded = static_cast<float>(nearT);
    nearDistance = std::max(ray.tmin, std::nextafter(rounded, -infinity));
    return true;
}

// Geometric triangle query with an existing Hit.t as an optional upper bound.
// Ray-space shear reduces it to signed 2D edge tests and barycentric interpolation.
bool intersectTriangle(const Triangle& triangle, const Ray& ray, Hit& hit) {
    if (!validRay(ray) || std::isnan(hit.t)) return false;
    const double limit = std::min(static_cast<double>(ray.tmax), hit.t);
    if (limit < ray.tmin) return false;

    // Ray-aligned edge functions are shared-edge consistent, unlike an epsilon
    // determinant test. Double intermediates also retain tiny/large triangles.
    int kz = 0;
    if (std::abs(ray.direction.y) > std::abs(ray.direction[kz])) kz = 1;
    if (std::abs(ray.direction.z) > std::abs(ray.direction[kz])) kz = 2;
    int kx = (kz+1)%3, ky = (kx+1)%3;
    if (ray.direction[kz] < 0) std::swap(kx, ky);
    const double sx = static_cast<double>(ray.direction[kx])/ray.direction[kz];
    const double sy = static_cast<double>(ray.direction[ky])/ray.direction[kz];
    const double sz = 1.0/ray.direction[kz];
    // Reassemble split origins before subtracting vertices; otherwise a precision
    // replay would lose exactly the low bits it is intended to preserve.
    const DVec3 origin = DVec3(ray.origin)+DVec3(ray.originLow);
    const DVec3 a = DVec3(triangle.v[0].position)-origin;
    const DVec3 b = DVec3(triangle.v[1].position)-origin;
    const DVec3 c = DVec3(triangle.v[2].position)-origin;
    const double ax = a[kx]-sx*a[kz], ay = a[ky]-sy*a[kz];
    const double bx = b[kx]-sx*b[kz], by = b[ky]-sy*b[kz];
    const double cx = c[kx]-sx*c[kz], cy = c[ky]-sy*c[kz];
    const double w0 = cx*by-cy*bx, w1 = ax*cy-ay*cx, w2 = bx*ay-by*ax;
    // Mixed edge signs mean outside the triangle; zero edges remain inclusive.
    if ((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) return false;
    const double determinant = w0+w1+w2;
    if (determinant == 0 || !std::isfinite(determinant)) return false;
    const double t = (w0*(sz*a[kz])+w1*(sz*b[kz])+w2*(sz*c[kz]))/determinant;
    if (!std::isfinite(t) || std::abs(t) > std::numeric_limits<float>::max() || t < ray.tmin || t > limit) return false;
    hit.t = t;
    hit.u = w1/determinant;
    hit.v = w2/determinant;
    return true;
}

// Bias along the GEOMETRIC normal on the outgoing side, never a normal-map normal.
// The bias scales with world coordinates and rounds outward by a representable step.
Ray offsetRay(Vec3 position, Vec3 geometricNormal, Vec3 direction, float maxDistance) {
    if (!finite(position) || !finite(geometricNormal) || !finite(direction) ||
        !nonzero(geometricNormal) || !nonzero(direction) || std::isnan(maxDistance) || maxDistance < 0)
        throw std::invalid_argument("Invalid secondary ray");
    direction = unit(direction);
    Vec3 n = unit(geometricNormal);
    if (preciseDot(DVec3(n), DVec3(direction)) < 0) n = -n;
    const double scale = std::max({std::abs(static_cast<double>(position.x)), std::abs(static_cast<double>(position.y)), std::abs(static_cast<double>(position.z))});
    const double distance = 4*std::numeric_limits<float>::epsilon()*scale;
    Vec3 origin = position;
    for (int axis = 0; axis < 3; ++axis) {
        origin[axis] = static_cast<float>(static_cast<double>(position[axis])+distance*n[axis]);
        if (n[axis] != 0) origin[axis] = std::nextafter(origin[axis], n[axis] > 0 ? infinity : -infinity);
    }
    if (!finite(origin)) throw std::overflow_error("Secondary ray offset exceeds finite coordinates");
    float limit = maxDistance;
    if (std::isfinite(limit)) {
        // Moving the start consumes part of a finite segment; stop short of its end.
        const double advance = preciseDot(DVec3(origin)-DVec3(position), DVec3(direction));
        limit = static_cast<float>(std::max(0.0, static_cast<double>(limit)-advance));
        limit = std::nextafter(limit, 0.0f);
    }
    return {origin, direction, 0, limit};
}

// Raster coordinates have a top-left origin. Re-orthogonalize the imported basis
// so node scale does not accidentally alter field of view or skew the image.
Ray Camera::generate(float pixelX, float pixelY, int width, int height) const {
    if (width <= 0 || height <= 0 || !std::isfinite(pixelX) || !std::isfinite(pixelY) ||
        !finite(position) || !finite(forward) || !finite(right) || !finite(up) ||
        !std::isfinite(yfov) || yfov <= 0 || yfov >= pi || !std::isfinite(aspect) || aspect < 0 ||
        !std::isfinite(nearClip) || nearClip < 0 || std::isnan(farClip) || farClip <= nearClip)
        throw std::invalid_argument("Invalid perspective camera or image dimensions");
    const Vec3 f = unit(forward);
    const Vec3 r = orthogonalized(DVec3(right), f);
    const Vec3 u = orthogonalized(DVec3(orthogonalized(DVec3(up),f)),r);
    if (!nonzero(f) || !nonzero(r) || !nonzero(u)) throw std::invalid_argument("Degenerate camera basis");
    const double ratio = aspect > 0 ? aspect : static_cast<double>(width)/height;
    const double halfHeight = std::tan(static_cast<double>(yfov)*0.5);
    const double x = (2*static_cast<double>(pixelX)/width-1)*ratio*halfHeight;
    const double y = (1-2*static_cast<double>(pixelY)/height)*halfHeight;
    const Vec3 direction = unit(DVec3(f)+DVec3(r)*x+DVec3(u)*y);
    // Convert camera-plane clip distances into parameters along this oblique ray.
    const double projection = preciseDot(DVec3(direction), DVec3(f));
    const double nearT = nearClip/projection;
    if (!(projection > 0) || !std::isfinite(nearT) || nearT > std::numeric_limits<float>::max())
        throw std::invalid_argument("Unrepresentable perspective ray");
    return {position, direction, static_cast<float>(nearT), static_cast<float>(farClip/projection)};
}

// Shared base-level texture sampler. Apply scale, then rotation, then translation;
// select the requested UV set, and respect separate S/T wrapping and filtering.
Vec4 Scene::sample(const TextureSlot& slot, Vec2 uv0, Vec2 uv1, bool srgb) const {
    // Missing maps are multiplicative identities for material factors.
    if (slot.texture == -1) return {1,1,1,1};
    if (slot.texture < 0 || static_cast<std::size_t>(slot.texture) >= textures.size())
        throw std::out_of_range("Texture slot index out of range");
    if (slot.texcoord < 0 || slot.texcoord > 1) throw std::out_of_range("Only TEXCOORD_0 and TEXCOORD_1 are supported");
    const Texture& texture = textures[static_cast<std::size_t>(slot.texture)];
    if (texture.image < 0 || static_cast<std::size_t>(texture.image) >= images.size())
        throw std::out_of_range("Texture image index out of range");
    const Image& image = images[static_cast<std::size_t>(texture.image)];
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.width)*static_cast<std::size_t>(image.height))
        throw std::runtime_error("Texture image dimensions do not match its pixels");
    const Vec2 uv = slot.texcoord == 0 ? uv0 : uv1;
    if (!finite2(uv) || !finite2(slot.offset) || !finite2(slot.scale) || !std::isfinite(slot.rotation))
        throw std::runtime_error("Non-finite texture coordinates or transform");
    const double c = std::cos(static_cast<double>(slot.rotation)), s = std::sin(static_cast<double>(slot.rotation));
    const double x = static_cast<double>(uv.x)*slot.scale.x, y = static_cast<double>(uv.y)*slot.scale.y;
    const double u = wrappedCoordinate(c*x-s*y+slot.offset.x, texture.wrapS);
    const double v = wrappedCoordinate(s*x+c*y+slot.offset.y, texture.wrapT);
    const auto texel = [&](int64_t ix, int64_t iy) {
        const int tx = wrappedIndex(ix, image.width, texture.wrapS), ty = wrappedIndex(iy, image.height, texture.wrapT);
        Vec4 value = image.pixels[static_cast<std::size_t>(ty)*static_cast<std::size_t>(image.width)+static_cast<std::size_t>(tx)];
        if (srgb) {
            value.x = srgbToLinear(value.x);
            value.y = srgbToLinear(value.y);
            value.z = srgbToLinear(value.z);
        }
        return value;
    };
    if (!texture.linear) return texel(static_cast<int64_t>(std::floor(u*image.width)), static_cast<int64_t>(std::floor(v*image.height)));
    const double px = u*image.width-0.5, py = v*image.height-0.5;
    const int64_t ix = static_cast<int64_t>(std::floor(px)), iy = static_cast<int64_t>(std::floor(py));
    const float fx = static_cast<float>(px-std::floor(px)), fy = static_cast<float>(py-std::floor(py));
    // Decode each texel, not the filtered color. Alpha is always linear.
    return interpolate(interpolate(texel(ix,iy),texel(ix+1,iy),fx),
                       interpolate(texel(ix,iy+1),texel(ix+1,iy+1),fx), fy);
}

// Material-level hit filtering is identical for camera, bounce, and shadow rays.
// Transmissive backfaces must remain visible so a closed dielectric can be exited.
bool Scene::accepts(const Triangle& triangle, const Ray& ray, const Hit& hit) const {
    if (triangle.material >= materials.size()) throw std::out_of_range("Triangle material index out of range");
    const Material& material = materials[triangle.material];
    if (!material.doubleSided && material.transmission <= 0) {
        Vec3 normal = triangle.geometricNormal;
        if (!nonzero(normal)) normal = unit(preciseCross(DVec3(triangle.v[1].position)-DVec3(triangle.v[0].position), DVec3(triangle.v[2].position)-DVec3(triangle.v[0].position)));
        if (preciseDot(DVec3(normal),DVec3(ray.direction)) >= 0) return false;
    }
    if (material.alphaMode == AlphaMode::Mask) {
        const Vec4 color = sample(material.baseColorTexture, interpolateUv(triangle,hit,0), interpolateUv(triangle,hit,1), false);
        return material.baseColor.w*color.w >= material.alphaCutoff;
    }
    return true;
}

// Validate references/numbers before workers use them, then prepare triangle area,
// normals, and tangents. Invalid geometry is diagnosed rather than quietly shaded.
void Scene::finalize() {
    for (std::size_t i = 0; i < images.size(); ++i) {
        const Image& image = images[i];
        if (image.width <= 0 || image.height <= 0 ||
            image.pixels.size() != static_cast<std::size_t>(image.width)*static_cast<std::size_t>(image.height))
            throw std::runtime_error("Image "+std::to_string(i)+" has invalid dimensions or pixel count");
        for (Vec4 pixel : image.pixels)
            if (!finite4(pixel)) throw std::runtime_error("Image "+std::to_string(i)+" contains non-finite pixels");
    }
    for (std::size_t i = 0; i < textures.size(); ++i) {
        const Texture& texture = textures[i];
        if (texture.image < 0 || static_cast<std::size_t>(texture.image) >= images.size())
            throw std::runtime_error("Texture "+std::to_string(i)+" has an invalid image index");
        const auto validWrap = [](Wrap w) { return w == Wrap::Repeat || w == Wrap::Clamp || w == Wrap::Mirror; };
        if (!validWrap(texture.wrapS) || !validWrap(texture.wrapT))
            throw std::runtime_error("Texture "+std::to_string(i)+" has an invalid wrap mode");
    }
    for (std::size_t i = 0; i < materials.size(); ++i) {
        const Material& material = materials[i];
        const std::string label = "Material "+std::to_string(i);
        const auto inUnit = [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; };
        if (!finite4(material.baseColor) || !inUnit(material.baseColor.x) || !inUnit(material.baseColor.y) ||
            !inUnit(material.baseColor.z) || !inUnit(material.baseColor.w) || !inUnit(material.metallic) ||
            !inUnit(material.roughness) || !inUnit(material.transmission) || !inUnit(material.alphaCutoff) ||
            !finite(material.emission) || material.emission.x < 0 || material.emission.y < 0 || material.emission.z < 0 ||
            !std::isfinite(material.ior) || material.ior <= 0 || !std::isfinite(material.thickness) || material.thickness < 0 ||
            std::isnan(material.attenuationDistance) || material.attenuationDistance <= 0 ||
            !inUnit(material.attenuationColor.x) || !inUnit(material.attenuationColor.y) || !inUnit(material.attenuationColor.z) ||
            !std::isfinite(material.normalScale))
            throw std::runtime_error(label+" has non-finite or out-of-range parameters");
        for (const TextureSlot* slot : {&material.baseColorTexture, &material.metallicRoughnessTexture, &material.normalTexture, &material.emissiveTexture, &material.transmissionTexture}) {
            if (slot->texture < -1 || (slot->texture >= 0 && static_cast<std::size_t>(slot->texture) >= textures.size()))
                throw std::runtime_error(label+" has an invalid texture index");
            if (slot->texture >= 0 && (slot->texcoord < 0 || slot->texcoord > 1 || !finite2(slot->offset) || !finite2(slot->scale) || !std::isfinite(slot->rotation)))
                throw std::runtime_error(label+" has invalid texture coordinates or transform");
        }
        if (material.normalTexture.texture >= 0 && (material.normalTexture.scale.x == 0 || material.normalTexture.scale.y == 0))
            warnings.push_back(label+" has a singular normal-map UV transform; using the untransformed tangent frame");
    }
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& triangle = triangles[i];
        if (triangle.material >= materials.size()) throw std::runtime_error("Triangle "+std::to_string(i)+" has an invalid material index");
        for (const Vertex& v : triangle.v)
            if (!finite(v.position) || !finite(v.normal) || !finite4(v.tangent) || !finite2(v.uv[0]) || !finite2(v.uv[1]))
                throw std::runtime_error("Triangle "+std::to_string(i)+" contains non-finite vertex data");
    }

    std::size_t kept = 0, degenerate = 0, firstDegenerate = 0, generatedNormals = 0, fallbackTangents = 0;
    // Compact only genuinely zero/unrepresentable-area triangles; an arbitrary
    // world-space epsilon would incorrectly delete thin or very small geometry.
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        Triangle& triangle = triangles[i];
        const DVec3 normal = preciseCross(DVec3(triangle.v[1].position)-DVec3(triangle.v[0].position), DVec3(triangle.v[2].position)-DVec3(triangle.v[0].position));
        const double twiceArea = std::sqrt(preciseDot(normal,normal));
        if (twiceArea*0.5 > std::numeric_limits<float>::max())
            throw std::runtime_error("Triangle "+std::to_string(i)+" area exceeds the supported finite range");
        triangle.area = static_cast<float>(twiceArea*0.5);
        if (!(triangle.area > 0)) {
            if (degenerate == 0) firstDegenerate = i;
            ++degenerate;
            continue;
        }
        triangle.geometricNormal = unit(normal);
        const Material& material = materials[triangle.material];
        const int uvSet = material.normalTexture.texture >= 0 ? material.normalTexture.texcoord : 0;
        DVec3 uvTangent{0,0,0}, uvBitangent{0,0,0};
        const bool usableUvs = uvDirections(triangle,uvSet,uvTangent,uvBitangent);
        for (Vertex& vertex : triangle.v) {
            // Keep vertex normals in the geometric hemisphere, preserving tangent
            // handedness when a normal must be flipped or reconstructed.
            Vec3 n = unit(vertex.normal);
            float handedness = vertex.tangent.w < 0 ? -1.0f : 1.0f;
            if (!nonzero(n) || dot(n,triangle.geometricNormal) == 0) {
                n = triangle.geometricNormal;
                ++generatedNormals;
            } else if (dot(n,triangle.geometricNormal) < 0) {
                n = -n;
                handedness = -handedness;
            }
            Vec3 tangent = orthogonalized(DVec3(Vec3{vertex.tangent.x,vertex.tangent.y,vertex.tangent.z}),n);
            if (!nonzero(tangent)) {
                // Prefer UV-derived tangents; an orthogonal basis is the last resort.
                if (usableUvs) tangent = orthogonalized(uvTangent,n);
                if (nonzero(tangent)) {
                    handedness = preciseDot(DVec3(cross(n,tangent)),uvBitangent) < 0 ? -1.0f : 1.0f;
                } else {
                    tangent = Frame(n).x;
                    handedness = 1;
                    if (material.normalTexture.texture >= 0) ++fallbackTangents;
                }
            }
            vertex.normal = n;
            vertex.tangent = {tangent.x,tangent.y,tangent.z,handedness};
        }
        if (kept != i) triangles[kept] = triangle;
        ++kept;
    }
    triangles.resize(kept);
    if (degenerate) warnings.push_back("Skipped "+std::to_string(degenerate)+" degenerate triangles (first input triangle "+std::to_string(firstDegenerate)+")");
    if (generatedNormals) warnings.push_back("Generated flat normals for "+std::to_string(generatedNormals)+" missing or tangential vertex normals");
    if (fallbackTangents) warnings.push_back("Used orthogonal fallback tangents for "+std::to_string(fallbackTangents)+" normal-mapped vertices with degenerate UVs");
}

// Turn a geometric hit into the local material state consumed by the BSDF.
// Vertex interpolation happens first; texture factors and normal mapping follow.
Surface surfaceAt(const Scene& scene, const Ray& ray, const Hit& hit) {
    if (hit.triangle >= scene.triangles.size() || !std::isfinite(hit.t) || !std::isfinite(hit.u) || !std::isfinite(hit.v))
        throw std::out_of_range("Invalid surface hit");
    const Triangle& triangle = scene.triangles[hit.triangle];
    if (triangle.material >= scene.materials.size()) throw std::out_of_range("Surface material index out of range");
    const Material& material = scene.materials[triangle.material];
    const double w = 1.0-static_cast<double>(hit.u)-hit.v;
    const DVec3 p = DVec3(triangle.v[0].position)*w+DVec3(triangle.v[1].position)*hit.u+DVec3(triangle.v[2].position)*hit.v;
    Surface surface;
    surface.position = {static_cast<float>(p.x),static_cast<float>(p.y),static_cast<float>(p.z)};
    // Retain the rounding residual for offset-free, high-precision path replay.
    surface.positionLow = {static_cast<float>(p.x-surface.position.x),static_cast<float>(p.y-surface.position.y),static_cast<float>(p.z-surface.position.z)};
    surface.geometricNormal = triangle.geometricNormal;
    surface.frontFace = preciseDot(DVec3(ray.direction),DVec3(surface.geometricNormal)) < 0;
    surface.normal = unit(DVec3(triangle.v[0].normal)*w+DVec3(triangle.v[1].normal)*hit.u+DVec3(triangle.v[2].normal)*hit.v, surface.geometricNormal);
    if (dot(surface.normal,surface.geometricNormal) < 0) surface.normal = -surface.normal;
    const auto tangentOf = [](const Vertex& v) { return DVec3(Vec3{v.tangent.x,v.tangent.y,v.tangent.z}); };
    surface.tangent = orthogonalized(tangentOf(triangle.v[0])*w+tangentOf(triangle.v[1])*hit.u+tangentOf(triangle.v[2])*hit.v, surface.normal, Frame(surface.normal).x);
    float handedness = w*triangle.v[0].tangent.w+static_cast<double>(hit.u)*triangle.v[1].tangent.w+static_cast<double>(hit.v)*triangle.v[2].tangent.w < 0 ? -1.0f : 1.0f;
    surface.bitangent = cross(surface.normal,surface.tangent)*handedness;
    const Vec2 uv0 = interpolateUv(triangle,hit,0), uv1 = interpolateUv(triangle,hit,1);
    const Vec4 base = scene.sample(material.baseColorTexture,uv0,uv1,true);
    surface.baseColor = Vec3{material.baseColor.x,material.baseColor.y,material.baseColor.z}*Vec3{base.x,base.y,base.z};
    const Vec4 mr = scene.sample(material.metallicRoughnessTexture,uv0,uv1,false);
    // glTF packs roughness in G and metallic in B; these are linear data channels.
    surface.metallic = std::clamp(material.metallic*mr.z,0.0f,1.0f);
    surface.roughness = std::clamp(material.roughness*mr.y,0.0f,1.0f);
    const Vec4 emissive = scene.sample(material.emissiveTexture,uv0,uv1,true);
    surface.emission = material.emission*Vec3{emissive.x,emissive.y,emissive.z};
    surface.transmission = std::clamp(material.transmission*scene.sample(material.transmissionTexture,uv0,uv1,false).x,0.0f,1.0f);
    surface.ior = material.ior;
    surface.thickness = material.thickness;
    surface.attenuationDistance = material.attenuationDistance;
    surface.attenuationColor = material.attenuationColor;
    surface.material = triangle.material;
    surface.doubleSided = material.doubleSided;
    if (material.normalTexture.texture >= 0) {
        const TextureSlot& slot = material.normalTexture;
        if (slot.scale.x != 0 && slot.scale.y != 0) {
            // Tangents are stored in the original UV frame, so finalization is
            // idempotent. Transform the frame along with normal-map coordinates.
            const double c = std::cos(static_cast<double>(slot.rotation)), s = std::sin(static_cast<double>(slot.rotation));
            const DVec3 t = DVec3(surface.tangent)*(c/slot.scale.x)-DVec3(surface.bitangent)*(s/slot.scale.y);
            surface.tangent = orthogonalized(t,surface.normal,surface.tangent);
            if ((slot.scale.x < 0) != (slot.scale.y < 0)) handedness = -handedness;
            surface.bitangent = cross(surface.normal,surface.tangent)*handedness;
        }
        const Vec4 texel = scene.sample(slot,uv0,uv1,false);
        // Decode tangent-space [0,1] normals into [-1,1], scale X/Y perturbation,
        // and transform through TBN before restoring a usable orthogonal frame.
        const DVec3 mapped = DVec3(surface.tangent)*((2.0*texel.x-1)*material.normalScale)+
            DVec3(surface.bitangent)*((2.0*texel.y-1)*material.normalScale)+DVec3(surface.normal)*(2.0*texel.z-1);
        surface.normal = unit(mapped,surface.normal);
        if (dot(surface.normal,surface.geometricNormal) < 0) surface.normal = -surface.normal;
        surface.tangent = orthogonalized(DVec3(surface.tangent),surface.normal,Frame(surface.normal).x);
        surface.bitangent = cross(surface.normal,surface.tangent)*handedness;
    }
    return surface;
}

// Rebuild immutable acceleration data from finalized world-space triangles.
// Leaves reorder indices, not the scene's triangle/material storage.
void Bvh::build(const Scene& scene) {
    nodes_.clear();
    indices_.clear();
    const std::size_t count = scene.triangles.size();
    if (count == 0) return;
    if (count > std::numeric_limits<uint32_t>::max()/2)
        throw std::length_error("Too many triangles for 32-bit BVH node indices");
    for (const Triangle& triangle : scene.triangles)
        for (const Vertex& vertex : triangle.v)
            if (!finite(vertex.position)) throw std::runtime_error("Cannot build BVH from non-finite geometry");
    indices_.resize(count);
    std::iota(indices_.begin(),indices_.end(),uint32_t{0});
    nodes_.reserve(count);
    nodes_.emplace_back();
    subdivide(scene,0,0,static_cast<uint32_t>(count),0);
}

// Evaluate a binned surface-area heuristic (SAH) on each centroid axis. Splitting
// is worthwhile only when child intersection cost plus traversal beats a leaf.
void Bvh::subdivide(const Scene& scene, uint32_t nodeIndex, uint32_t begin, uint32_t end, int depth) {
    Bounds bounds, centers;
    for (uint32_t i = begin; i < end; ++i) {
        const Triangle& triangle = scene.triangles[indices_[i]];
        bounds.expand(triangle.bounds());
        centers.expand(centroid(triangle));
    }
    nodes_[nodeIndex] = {bounds,begin,end-begin};
    if (end-begin <= leafSize || depth >= maxBvhDepth) return;
    const double parentArea = boundsArea(bounds);
    if (!(parentArea > 0)) return;
    struct Bin { Bounds bounds; uint32_t count = 0; };
    std::array<std::array<Bin,binCount>,3> bins{};
    // Map centroids to bounded bins; coincident centroids cannot provide a split.
    const auto binAt = [&](float center, int axis) {
        const double extent = static_cast<double>(centers.hi[axis])-centers.lo[axis];
        if (!(extent > 0)) return 0;
        return std::clamp(static_cast<int>((static_cast<double>(center)-centers.lo[axis])/extent*binCount),0,binCount-1);
    };
    for (uint32_t i = begin; i < end; ++i) {
        const Triangle& triangle = scene.triangles[indices_[i]];
        const Bounds box = triangle.bounds();
        const Vec3 center = centroid(triangle);
        for (int axis = 0; axis < 3; ++axis) {
            Bin& bin = bins[axis][binAt(center[axis],axis)];
            bin.bounds.expand(box);
            ++bin.count;
        }
    }
    int bestAxis = -1, bestSplit = -1;
    double bestCost = static_cast<double>(end-begin)*parentArea;
    for (int axis = 0; axis < 3; ++axis) {
        // Right-to-left suffix bounds/counts make every possible bin split cheap
        // to evaluate during the subsequent left-to-right prefix scan.
        if (centers.lo[axis] == centers.hi[axis]) continue;
        std::array<Bounds,binCount> rightBounds{};
        std::array<uint32_t,binCount> rightCount{};
        Bounds right;
        uint32_t count = 0;
        for (int b = binCount-1; b >= 0; --b) {
            if (bins[axis][b].count) right.expand(bins[axis][b].bounds);
            count += bins[axis][b].count;
            rightBounds[b] = right;
            rightCount[b] = count;
        }
        Bounds left;
        count = 0;
        for (int b = 0; b < binCount-1; ++b) {
            if (bins[axis][b].count) left.expand(bins[axis][b].bounds);
            count += bins[axis][b].count;
            if (!count || !rightCount[b+1]) continue;
            const double cost = parentArea+count*boundsArea(left)+rightCount[b+1]*boundsArea(rightBounds[b+1]);
            if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = b; }
        }
    }
    if (bestAxis < 0) return;
    // Partition the index range in place, then allocate adjacent left/right nodes.
    const auto middle = std::partition(indices_.begin()+begin,indices_.begin()+end,[&](uint32_t index) {
        return binAt(centroid(scene.triangles[index])[bestAxis],bestAxis) <= bestSplit;
    });
    const uint32_t mid = static_cast<uint32_t>(middle-indices_.begin());
    if (mid == begin || mid == end) return;
    const uint32_t firstChild = static_cast<uint32_t>(nodes_.size());
    nodes_.resize(nodes_.size()+2);
    nodes_[nodeIndex].first = firstChild;
    nodes_[nodeIndex].count = 0;
    subdivide(scene,firstChild,begin,mid,depth+1);
    subdivide(scene,firstChild+1,mid,end,depth+1);
}

// Iterative traversal shares alpha/culling rules between closest and any-hit
// queries. A bounded stack replaces recursion in this per-ray hot path.
bool Bvh::intersect(const Scene& scene, const Ray& ray, Hit& hit, bool any, uint32_t ignoreTriangle) const {
    if (nodes_.empty() || !validRay(ray) || std::isnan(hit.t)) return false;
    Ray clipped = ray;
    double closestLimit=std::min(static_cast<double>(ray.tmax),hit.t);
    clipped.tmax = std::min(clipped.tmax,std::nextafter(static_cast<float>(closestLimit),infinity));
    const float queryLimit = clipped.tmax;
    if (queryLimit < ray.tmin) return false;
    std::array<double,3> inverse{};
    // Precompute once per ray instead of dividing again for every visited box.
    const DVec3 preciseOrigin=DVec3(ray.origin)+DVec3(ray.originLow);
    for (int axis=0;axis<3;++axis) if (ray.direction[axis]!=0) inverse[axis]=1.0/ray.direction[axis];
    const auto intersectsBox=[&](const Bounds& box, float& nearDistance) {
        double nearT=clipped.tmin, farT=clipped.tmax;
        for (int axis=0;axis<3;++axis) {
            const double origin=preciseOrigin[axis];
            if (ray.direction[axis]==0) {
                if (origin<box.lo[axis] || origin>box.hi[axis]) return false;
                continue;
            }
            double a=(static_cast<double>(box.lo[axis])-origin)*inverse[axis];
            double b=(static_cast<double>(box.hi[axis])-origin)*inverse[axis];
            if (inverse[axis]<0) std::swap(a,b);
            // Input bounds are finite and validated at build time. This bounds
            // subtraction/reciprocal/multiply error without per-node divisions
            // or six nextafter calls; leave the public scalar slab test intact.
            constexpr double error=4*std::numeric_limits<double>::epsilon();
            a-=std::abs(a)*error; b+=std::abs(b)*error;
            nearT=std::max(nearT,a); farT=std::min(farT,b);
            if (nearT>farT) return false;
        }
        nearDistance=std::max(clipped.tmin,std::nextafter(static_cast<float>(nearT),-infinity));
        return true;
    };
    float rootNear = 0;
    if (!intersectsBox(nodes_[0].box,rootNear)) return false;
    struct Entry { uint32_t node; float near; };
    std::array<Entry,maxBvhDepth+2> stack{};
    std::size_t size = 0;
    stack[size++] = {0,rootNear};
    bool found = false;
    while (size) {
        const Entry entry = stack[--size];
        if (entry.near > clipped.tmax) continue;
        const Node& node = nodes_[entry.node];
        if (node.count) {
            // A leaf tests its actual triangles. Skip only the requested source
            // primitive; all neighboring interfaces remain candidates.
            for (uint32_t i = node.first; i < node.first+node.count; ++i) {
                const uint32_t index = indices_[i];
                if (index==ignoreTriangle) continue;
                Hit candidate;
                candidate.t = closestLimit;
                candidate.triangle = index;
                if (!intersectTriangle(scene.triangles[index],clipped,candidate) || !scene.accepts(scene.triangles[index],clipped,candidate)) continue;
                if (any) { hit = candidate; return true; }
                if (!found || candidate.t < hit.t || (candidate.t == hit.t && index < hit.triangle)) {
                    hit = candidate;
                    closestLimit=candidate.t;
                    // Traversal bounds stay conservative floats, while hit
                    // ordering and reconstruction retain double precision.
                    clipped.tmax = std::min(queryLimit,std::nextafter(static_cast<float>(candidate.t),infinity));
                    found = true;
                }
            }
        } else {
            float leftNear = 0, rightNear = 0;
            const bool left = intersectsBox(nodes_[node.first].box,leftNear);
            const bool right = intersectsBox(nodes_[node.first+1].box,rightNear);
            if (size+static_cast<std::size_t>(left)+static_cast<std::size_t>(right) > stack.size())
                throw std::logic_error("BVH traversal exceeded its bounded stack");
            if (left && right) {
                // LIFO order: push the farther child first so the nearer one can
                // tighten the closest limit before the farther child is revisited.
                if (leftNear <= rightNear) {
                    stack[size++] = {node.first+1,rightNear};
                    stack[size++] = {node.first,leftNear};
                } else {
                    stack[size++] = {node.first,leftNear};
                    stack[size++] = {node.first+1,rightNear};
                }
            } else if (left) stack[size++] = {node.first,leftNear};
            else if (right) stack[size++] = {node.first+1,rightNear};
        }
    }
    return found;
}

// Capacity, rather than element count, reflects the memory actually reserved.
std::size_t Bvh::bytes() const {
    return nodes_.capacity()*sizeof(Node)+indices_.capacity()*sizeof(uint32_t);
}
}
