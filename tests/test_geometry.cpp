// Analytic geometry/sampling checks plus brute-force parity for the accelerator.
// Failures throw in every build configuration, including optimized Release.
#include "scene.h"

#include <chrono>
#include <iostream>
#include <string>

namespace {
using namespace cr;
std::size_t checks = 0;

// Count explicit assertions and compare finite values with scale-aware tolerances.
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void near(float actual, float expected, const std::string& message, float absolute = 1e-6f, float relative = 2e-5f) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(static_cast<double>(actual)-expected) <= absolute+static_cast<double>(relative)*std::max(std::abs(actual),std::abs(expected)),
        message+": actual="+std::to_string(actual)+", expected="+std::to_string(expected));
}
void near(Vec3 actual, Vec3 expected, const std::string& message, float tolerance = 2e-5f) {
    near(actual.x,expected.x,message+" x",tolerance);
    near(actual.y,expected.y,message+" y",tolerance);
    near(actual.z,expected.z,message+" z",tolerance);
}
void near(Vec4 actual, Vec4 expected, const std::string& message) {
    near(Vec3{actual.x,actual.y,actual.z},Vec3{expected.x,expected.y,expected.z},message);
    near(actual.w,expected.w,message+" alpha");
}
// Negative tests must throw, optionally with a useful diagnostic substring.
template<class F> void throws(F action, const std::string& label, const std::string& diagnostic = {}) {
    try { action(); }
    catch (const std::exception& error) {
        require(diagnostic.empty() || std::string(error.what()).find(diagnostic) != std::string::npos,label+" diagnostic: "+error.what());
        return;
    }
    require(false,label+" did not throw");
}
// Compact fixture factories isolate geometry tests from files and the glTF adapter.
Triangle triangle(Vec3 a = {0,0,0}, Vec3 b = {1,0,0}, Vec3 c = {0,1,0}, uint32_t material = 0) {
    Triangle t;
    t.v[0].position = a; t.v[1].position = b; t.v[2].position = c;
    t.v[0].uv[0] = {0,0}; t.v[1].uv[0] = {1,0}; t.v[2].uv[0] = {0,1};
    for (Vertex& v : t.v) v.uv[1] = v.uv[0];
    t.material = material;
    return t;
}
Scene singleTriangle(Triangle t = triangle()) {
    Scene scene;
    scene.materials.emplace_back();
    scene.triangles.push_back(t);
    return scene;
}
int addTexture(Scene& scene, Image image, bool linear = false, Wrap wrap = Wrap::Clamp) {
    const int imageIndex = static_cast<int>(scene.images.size());
    scene.images.push_back(std::move(image));
    scene.textures.push_back({imageIndex,wrap,wrap,linear});
    return static_cast<int>(scene.textures.size()-1);
}
int solidTexture(Scene& scene, Vec4 color) { return addTexture(scene,{1,1,{color}}); }
// TBN must be finite, orthonormal, and have the expected mirrored/non-mirrored sign.
void checkFrame(const Surface& surface, float handedness, const std::string& label) {
    require(finite(surface.normal) && finite(surface.tangent) && finite(surface.bitangent),label+" finite frame");
    near(length(surface.normal),1,label+" unit normal");
    near(length(surface.tangent),1,label+" unit tangent");
    near(length(surface.bitangent),1,label+" unit bitangent");
    near(dot(surface.normal,surface.tangent),0,label+" N dot T");
    near(dot(surface.normal,surface.bitangent),0,label+" N dot B");
    near(dot(surface.tangent,surface.bitangent),0,label+" T dot B");
    near(dot(cross(surface.normal,surface.tangent),surface.bitangent),handedness,label+" handedness");
}

// Known hits/misses cover shared edges, inclusive intervals, signed zero,
// degeneracy, grazing directions, invalid inputs, and very small/large scales.
void testIntersections() {
    const Triangle t = triangle();
    const Ray ray{{0.25f,0.25f,1},{0,0,-1}};
    Hit hit;
    require(intersectTriangle(t,ray,hit),"triangle center");
    near(hit.t,1,"triangle distance"); near(hit.u,0.25f,"triangle u"); near(hit.v,0.25f,"triangle v");
    for (Vec3 p : {Vec3{0,0,1},Vec3{1,0,1},Vec3{0,1,1},Vec3{0.5f,0.5f,1},Vec3{0,0.5f,1}}) {
        Hit edge;
        require(intersectTriangle(t,{p,{0,0,-1}},edge),"inclusive triangle edge/vertex");
    }
    Hit inclusive;
    require(intersectTriangle(t,{ray.origin,ray.direction,1,1},inclusive),"closed triangle interval");
    for (Ray miss : {Ray{ray.origin,ray.direction,0,std::nextafter(1.0f,0.0f)},
                     Ray{ray.origin,ray.direction,std::nextafter(1.0f,2.0f),2},
                     Ray{ray.origin,{1,0,0}}, Ray{{0.2f,0.2f,0},{1,0,0}},
                     Ray{ray.origin,{}}, Ray{{1,1,1},{0,0,-1}}, Ray{ray.origin,ray.direction,2,1}}) {
        Hit unchanged{7,0.3f,0.4f,99};
        require(!intersectTriangle(t,miss,unchanged),"triangle miss or interval rejection");
        near(unchanged.t,7,"failed triangle query preserves hit");
        require(unchanged.triangle == 99,"failed triangle query preserves index");
    }
    Hit bound{0.5f};
    require(!intersectTriangle(t,ray,bound),"existing Hit.t is an upper bound");
    Hit back;
    require(intersectTriangle(t,{{0.25f,0.25f,-1},{0,0,1}},back),"raw triangles are two-sided");
    Hit unnormalized;
    require(intersectTriangle(t,{ray.origin,{0,0,-2}},unnormalized),"non-unit triangle direction");
    near(unnormalized.t,0.5f,"non-unit triangle parameter");
    Hit grazing;
    require(intersectTriangle(t,{{-0.75f,0.25f,1e-12f},{1,0,-1e-12f}},grazing),"near-parallel triangle ray");
    near(grazing.t,1,"near-parallel triangle distance");
    Hit degenerate;
    require(!intersectTriangle(triangle({0,0,0},{1,0,0},{2,0,0}),ray,degenerate),"collinear triangle");
    for (float scale : {1e-20f,1e-10f,1.0f,1e10f}) {
        Hit scaled;
        const Triangle scaledTriangle = triangle({0,0,0},{scale,0,0},{0,scale,0});
        require(intersectTriangle(scaledTriangle,{{scale*0.25f,scale*0.25f,scale},{0,0,-1}},scaled),"scale-independent triangle hit");
        near(scaled.t,scale,"scaled distance",0,2e-6f);
    }
    const Triangle other = triangle({1,1,0},{0,1,0},{1,0,0});
    for (int i = 0; i <= 100; ++i) {
        const float x = static_cast<float>(i)/100;
        const Ray edgeRay{{x,1-x,1},{0,0,-1}};
        Hit a,b;
        require(intersectTriangle(t,edgeRay,a) || intersectTriangle(other,edgeRay,b),"shared edge must not crack");
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (Ray invalid : {Ray{{nan,0,1},{0,0,-1}},Ray{ray.origin,{0,nan,-1}},Ray{ray.origin,ray.direction,nan,2},Ray{ray.origin,ray.direction,0,nan},Ray{{infinity,0,1},{0,0,-1}}}) {
        Hit invalidHit;
        require(!intersectTriangle(t,invalid,invalidHit),"non-finite triangle ray rejection");
    }
    Triangle invalidTriangle = t;
    invalidTriangle.v[1].position.x = nan;
    Hit invalidHit;
    require(!intersectTriangle(invalidTriangle,ray,invalidHit),"non-finite triangle rejection");

    Bounds box;
    // Slab tests separately verify parallel axes and flat boxes around triangles.
    box.expand({0,0,0}); box.expand({1,1,1});
    float nearDistance = -1;
    require(box.intersect({{0,0.5f,2},{-0.0f,0,-1}},nearDistance),"parallel slab boundary and signed zero");
    near(nearDistance,1,"slab entry");
    require(box.intersect({{0.5f,0.5f,0.5f},{1,0,0}},nearDistance),"ray starts inside bounds");
    near(nearDistance,0,"inside slab near distance");
    require(!box.intersect({{-0.1f,0.5f,2},{0,0,-1}},nearDistance),"parallel ray outside slab");
    require(!box.intersect({{0.5f,0.5f,2},{0,0,-1},0,0.5f},nearDistance),"bounds interval clipping");
    require(box.intersect({{0.5f,0.5f,2},{0,0,-1},1,1},nearDistance),"bounds endpoint inclusion");
    require(!box.intersect({{0.5f,0.5f,0.5f},{}},nearDistance),"zero direction bounds rejection");
    require(!Bounds{}.intersect(ray,nearDistance),"empty bounds rejection");
    Bounds flat = t.bounds();
    require(flat.intersect(ray,nearDistance),"flat triangle bounds");
    near(nearDistance,1,"flat bounds distance");
    require(!box.intersect({{nan,0,0},{1,0,0}},nearDistance),"NaN bounds ray rejection");
    Bounds tiny;
    const float denormal = std::numeric_limits<float>::denorm_min();
    tiny.expand({denormal,0,0}); tiny.expand({2*denormal,1,1});
    require(tiny.intersect({{0,0.5f,0.5f},{denormal,0,0},0,2},nearDistance),"subnormal direction without reciprocal overflow");
    near(nearDistance,1,"subnormal slab entry");
}

// Secondary biases must remove self hits without erasing nearby geometry.
// Camera checks distinguish perspective clip planes from spherical ray distances.
void testOffsetsAndCamera() {
    const Ray zero = offsetRay({0,0,0},{0,0,1},{0,0,1});
    require(zero.origin.z > 0 && finite(zero.origin),"origin offset rounds away from zero");
    Hit self;
    require(!intersectTriangle(triangle(),zero,self),"zero-coordinate self-intersection avoided");
    for (float scale : {1e-12f,1.0f,1e12f}) {
        const Vec3 position{0.25f*scale,0.25f*scale,scale};
        const Ray positive = offsetRay(position,{0,0,1},{0,0,2},0.01f*scale);
        const Ray negative = offsetRay(position,{0,0,1},{0,0,-1});
        require(positive.origin.z > position.z && negative.origin.z < position.z,"offset chooses outgoing geometric side");
        near(length(positive.direction),1,"secondary direction is normalized");
        require(positive.tmax < 0.01f*scale && positive.tmax > 0,"finite shadow endpoint shortened");
        const float nextZ = scale+0.001f*scale;
        Hit neighbor;
        require(intersectTriangle(triangle({0,0,nextZ},{scale,0,nextZ},{0,scale,nextZ}),positive,neighbor),"scale-aware offset preserves nearby surfaces");
        Hit same;
        require(!intersectTriangle(triangle({0,0,scale},{scale,0,scale},{0,scale,scale}),positive,same),"scale-aware offset prevents self hits");
    }
    throws([] { offsetRay({}, {}, {0,0,1}); },"zero offset normal");
    throws([] { offsetRay({}, {0,0,1}, {}, 1); },"zero offset direction");
    throws([] { offsetRay({}, {0,0,1}, {0,0,1}, -1); },"negative offset distance");

    Camera camera;
    camera.yfov = pi/2; camera.nearClip = 1; camera.farClip = 10;
    Ray center = camera.generate(100,50,200,100);
    near(center.direction,{0,0,-1},"camera center");
    near(center.tmin,1,"center near clip"); near(center.tmax,10,"center far clip");
    const Ray topLeft = camera.generate(0,0,200,100);
    near(topLeft.direction,normalize(Vec3{-2,1,-1}),"top-left image and vertical field of view");
    near(-topLeft.direction.z*topLeft.tmin,1,"near clip is a plane, not ray radius");
    near(-topLeft.direction.z*topLeft.tmax,10,"far clip is a plane, not ray radius");
    camera.aspect = 1;
    near(camera.generate(0,0,200,100).direction,normalize(Vec3{-1,1,-1}),"explicit glTF aspect overrides image aspect");
    camera.aspect = 0;
    camera.right = {2,0,0}; camera.up = {0,3,0}; camera.forward = {0,0,-4};
    near(camera.generate(0,0,200,100).direction,topLeft.direction,"camera basis scale does not change FOV");
    camera.position = {4,5,6}; camera.forward = {1,0,0}; camera.right = {0,0,1}; camera.up = {0,1,0};
    const Ray rotated = camera.generate(100,50,200,100);
    near(rotated.origin,camera.position,"camera world position"); near(rotated.direction,{1,0,0},"camera world rotation");
    camera.farClip = infinity;
    require(std::isinf(camera.generate(100,50,200,100).tmax),"infinite camera far plane");
    throws([&] { camera.generate(0,0,0,100); },"zero image width");
    camera.yfov = 0;
    throws([&] { camera.generate(0,0,100,100); },"invalid perspective angle");
    camera.yfov = pi/2; camera.farClip = 0.5f;
    throws([&] { camera.generate(0,0,100,100); },"reversed camera clipping");
    camera.farClip = 10; camera.right = camera.forward;
    throws([&] { camera.generate(0,0,100,100); },"singular camera frame");
    camera.forward = {1,1,1}; camera.right = camera.forward;
    throws([&] { camera.generate(0,0,100,100); },"oblique singular camera frame");
    camera.right = {1,-1,0}; camera.up = camera.forward;
    throws([&] { camera.generate(0,0,100,100); },"oblique singular camera up vector");
}

// Asymmetric tiny images expose flipped UVs, incorrect wraps/transform order,
// and gamma mistakes that uniform-color texture tests would not reveal.
void testTextureSampling() {
    Scene scene;
    TextureSlot slot;
    slot.texture = addTexture(scene,{2,2,{{1,0,0,0.2f},{0,1,0,0.4f},{0,0,1,0.6f},{1,1,1,0.8f}}});
    near(scene.sample(slot,{0.25f,0.25f},{},false),{1,0,0,0.2f},"top-left texel");
    near(scene.sample(slot,{0.75f,0.25f},{},false),{0,1,0,0.4f},"top-right texel");
    near(scene.sample(slot,{0.25f,0.75f},{},false),{0,0,1,0.6f},"bottom-left texel, no V flip");
    slot.texcoord = 1;
    near(scene.sample(slot,{0.25f,0.75f},{0.75f,0.25f},false),{0,1,0,0.4f},"TEXCOORD_1 selection");
    slot.texcoord = 0; slot.scale = {0.5f,1}; slot.rotation = pi/2; slot.offset = {1,0};
    near(scene.sample(slot,{0.4f,0.6f},{},false),{1,0,0,0.2f},"UV transform order scale, rotation, offset");
    slot.scale = {-1,1}; slot.rotation = 0; slot.offset = {1,0};
    near(scene.sample(slot,{0.25f,0.25f},{},false),{0,1,0,0.4f},"negative UV scale");
    slot.scale = {1,1}; slot.offset = {};
    Texture& texture = scene.textures[0];
    texture.wrapS = Wrap::Repeat;
    near(scene.sample(slot,{-0.25f,0.25f},{},false),{0,1,0,0.4f},"negative repeat U");
    near(scene.sample(slot,{1.25f,0.25f},{},false),{1,0,0,0.2f},"positive repeat U");
    near(scene.sample(slot,{1,0.25f},{},false),{1,0,0,0.2f},"repeat exact upper edge");
    texture.wrapS = Wrap::Mirror;
    near(scene.sample(slot,{-0.25f,0.25f},{},false),{1,0,0,0.2f},"negative mirror U");
    near(scene.sample(slot,{1.25f,0.25f},{},false),{0,1,0,0.4f},"odd mirror U");
    near(scene.sample(slot,{2.25f,0.25f},{},false),{1,0,0,0.2f},"even mirror U");
    near(scene.sample(slot,{1,0.25f},{},false),{0,1,0,0.4f},"mirror exact upper edge");
    texture.wrapS = Wrap::Clamp;
    near(scene.sample(slot,{-100,0.25f},{},false),{1,0,0,0.2f},"clamped lower U");
    near(scene.sample(slot,{100,0.25f},{},false),{0,1,0,0.4f},"clamped upper U");
    texture.wrapT = Wrap::Repeat;
    near(scene.sample(slot,{0.25f,-0.25f},{},false),{0,0,1,0.6f},"independent repeat V");
    texture.wrapT = Wrap::Mirror;
    near(scene.sample(slot,{0.25f,-0.25f},{},false),{1,0,0,0.2f},"independent mirror V");
    texture.wrapT = Wrap::Clamp;
    texture.linear = true;
    near(scene.sample(slot,{0.5f,0.5f},{},true),{0.5f,0.5f,0.5f,0.5f},"bilinear decoding happens before filtering");
    texture.wrapS = Wrap::Repeat;
    near(scene.sample(slot,{0,0.25f},{},false),{0.5f,0.5f,0,0.3f},"bilinear repeat seam");
    texture.wrapS = Wrap::Clamp;
    near(scene.sample(slot,{0,0.25f},{},false),{1,0,0,0.2f},"bilinear clamped edge");
    texture.wrapS = Wrap::Mirror;
    near(scene.sample(slot,{0,0.25f},{},false),{1,0,0,0.2f},"bilinear mirrored edge");
    near(scene.sample(slot,{1,0.25f},{},false),{0,1,0,0.4f},"bilinear mirrored upper edge");
    Scene colors;
    // Nontrivial midtones distinguish decode-before-filter from decode-after-filter.
    TextureSlot colorSlot;
    colorSlot.texture = addTexture(colors,{2,1,{{0.2f,0.4f,0.6f,0.1f},{0.8f,0.6f,0.4f,0.9f}}},true);
    const Vec4 filtered = colors.sample(colorSlot,{0.5f,0.5f},{},true);
    near(filtered.x,(srgbToLinear(0.2f)+srgbToLinear(0.8f))*0.5f,"nontrivial sRGB interpolation");
    near(filtered.y,(srgbToLinear(0.4f)+srgbToLinear(0.6f))*0.5f,"nontrivial sRGB green interpolation");
    near(filtered.w,0.5f,"alpha is not sRGB decoded");
    near(colors.sample(colorSlot,{0.5f,0.5f},{},false),{0.5f,0.5f,0.5f,0.5f},"linear data-map filtering");
    require(std::abs(filtered.x-srgbToLinear(0.5f)) > 0.05f,"reject decode-after-filter implementation");
    near(scene.sample(TextureSlot{},{},{},true),{1,1,1,1},"absent map is multiplicative identity");
    slot.texcoord = 2;
    throws([&] { scene.sample(slot,{},{},false); },"unsupported UV set");
    slot.texcoord = 0; slot.texture = 99;
    throws([&] { scene.sample(slot,{},{},false); },"invalid texture reference");
    slot.texture = 0; slot.scale = {1e30f,1e30f};
    require(std::isfinite(scene.sample(slot,{1e30f,-1e30f},{},false).x),"large finite UVs do not overflow integer indexing");
}

// Normal transforms must remain perpendicular under scale/shear, and both node
// mirroring and UV transforms must preserve the intended tangent-map orientation.
void testNormalsAndSurfaces() {
    Mat4 tiny;
    tiny.m[0]=tiny.m[5]=tiny.m[10]=1e-7f;
    near(tiny.normal({0,0,1}),{0,0,1},"small well-conditioned normal transform");
    Mat4 extreme;
    extreme.m[0]=1e-20f; extreme.m[5]=1e20f;
    near(extreme.normal({1,0,1}),{1,0,1e-20f},"normal transform avoids float overflow");
    Mat4 nonuniform;
    nonuniform.m[0] = 2; nonuniform.m[5] = 3; nonuniform.m[10] = 0.5f;
    const Vec3 n = normalize(Vec3{1,1,1});
    const Vec3 transformed = nonuniform.normal(n);
    near(transformed,normalize(Vec3{0.5f,1.0f/3,2}),"inverse-transpose nonuniform normal");
    near(dot(transformed,nonuniform.vector({1,-1,0})),0,"normal remains perpendicular after nonuniform scale");
    Mat4 shear = nonuniform;
    shear.m[4] = 0.7f; shear.m[8] = -0.3f; shear.m[9] = 0.2f;
    near(dot(shear.normal(n),shear.vector({1,-1,0})),0,"inverse transpose with shear");
    near(dot(shear.normal(n),shear.vector({1,0,-1})),0,"inverse transpose second tangent");
    Mat4 singular;
    singular.m[0] = 0;
    throws([&] { singular.normal({0,0,1}); },"singular transform diagnostic","Singular");

    Mat4 mirror = nonuniform;
    mirror.m[0] = -2;
    require(mirror.determinant() < 0,"mirrored transform sign");
    Triangle mirrored = triangle();
    for (Vertex& vertex : mirrored.v) {
        vertex.position = mirror.point(vertex.position);
        vertex.normal = mirror.normal({0,0,1});
        const Vec3 tangent = normalize(mirror.vector({1,0,0}));
        vertex.tangent = {tangent.x,tangent.y,tangent.z,-1};
    }
    // The loader compensates reflected winding and tangent handedness together.
    std::swap(mirrored.v[1],mirrored.v[2]);
    Scene scene = singleTriangle(mirrored);
    scene.materials[0].doubleSided = true;
    scene.materials[0].normalTexture.texture = solidTexture(scene,{1,0.5f,1,1});
    scene.finalize();
    const Ray front{{-0.5f,0.75f,1},{0,0,-1}}, back{{-0.5f,0.75f,-1},{0,0,1}};
    Hit hit;
    require(intersectTriangle(scene.triangles[0],front,hit),"mirrored transformed triangle hit");
    require(scene.accepts(scene.triangles[0],front,hit),"mirrored outward triangle acceptance");
    const Surface surface = surfaceAt(scene,front,hit);
    near(surface.geometricNormal,{0,0,1},"mirrored geometric winding compensated");
    near(surface.normal,normalize(Vec3{-1,0,1}),"mirrored tangent-space normal");
    require(surface.frontFace,"mirrored frontFace");
    checkFrame(surface,-1,"mirrored normal map");
    Hit backHit;
    require(intersectTriangle(scene.triangles[0],back,backHit),"mirrored back hit");
    const Surface backSurface = surfaceAt(scene,back,backHit);
    require(!backSurface.frontFace,"back hit preserves geometric frontFace");
    near(backSurface.geometricNormal,surface.geometricNormal,"back hit does not flip geometric normal");
    near(backSurface.normal,surface.normal,"outward shading normal contract on double-sided back hits");
    checkFrame(backSurface,-1,"double-sided mirrored normal map");

    Scene generated = singleTriangle();
    // Missing tangents should use the UV set of the normal map, not always UV0.
    generated.materials[0].normalTexture.texture = solidTexture(generated,{1,0.5f,1,1});
    generated.materials[0].normalTexture.texcoord = 1;
    generated.triangles[0].v[1].uv[1] = {0,1};
    generated.triangles[0].v[2].uv[1] = {1,0};
    generated.finalize();
    Hit generatedHit{1,0.25f,0.25f,0};
    Surface generatedSurface = surfaceAt(generated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit);
    near(generatedSurface.normal,normalize(Vec3{0,1,1}),"generated tangent uses normal-map UV1");
    checkFrame(generatedSurface,-1,"UV1 handedness");
    generated.materials[0].normalTexture.rotation = pi/2;
    generatedSurface = surfaceAt(generated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit);
    near(generatedSurface.normal,normalize(Vec3{-1,0,1}),"normal-map rotation transforms tangent frame");
    generated.materials[0].normalTexture.rotation = 0;
    generated.materials[0].normalTexture.scale = {-1,1};
    generatedSurface = surfaceAt(generated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit);
    near(generatedSurface.normal,normalize(Vec3{0,-1,1}),"normal-map negative scale transforms tangent frame");
    checkFrame(generatedSurface,1,"UV transform changes handedness");
    generated.finalize();
    near(surfaceAt(generated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit).normal,generatedSurface.normal,"repeated finalize does not reapply UV transform");
    generated.materials[0].normalScale = 0;
    near(surfaceAt(generated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit).normal,{0,0,1},"zero normal scale");

    Scene maps = singleTriangle();
    // Packed channels, color spaces, factors, and volume parameters reach Surface intact.
    Material& material = maps.materials[0];
    material.baseColor = {0.8f,0.6f,0.4f,0.5f}; material.metallic = 0.7f; material.roughness = 0.5f;
    material.transmission = 0.9f; material.emission = {2,3,4};
    material.ior = 1.33f; material.thickness = 2; material.attenuationDistance = 3; material.attenuationColor = {0.2f,0.4f,0.8f};
    material.baseColorTexture.texture = solidTexture(maps,{0.5f,0.25f,0.75f,0.2f});
    material.metallicRoughnessTexture.texture = solidTexture(maps,{0.1f,0.2f,0.8f,1});
    material.emissiveTexture.texture = solidTexture(maps,{0.6f,0.7f,0.8f,1});
    material.transmissionTexture.texture = solidTexture(maps,{0.4f,1,1,1});
    maps.finalize();
    const Surface mapped = surfaceAt(maps,{{0.25f,0.25f,1},{0,0,-1}},generatedHit);
    near(mapped.position,{0.25f,0.25f,0},"barycentric world position");
    near(mapped.baseColor,{0.8f*srgbToLinear(0.5f),0.6f*srgbToLinear(0.25f),0.4f*srgbToLinear(0.75f)},"base-color map sRGB factors");
    near(mapped.emission,{2*srgbToLinear(0.6f),3*srgbToLinear(0.7f),4*srgbToLinear(0.8f)},"emissive map sRGB factors");
    near(mapped.metallic,0.56f,"metallic blue channel"); near(mapped.roughness,0.1f,"roughness green channel");
    near(mapped.transmission,0.36f,"linear transmission red channel");
    near(mapped.ior,1.33f,"surface IOR"); near(mapped.thickness,2,"surface thickness");
    near(mapped.attenuationDistance,3,"surface attenuation distance"); near(mapped.attenuationColor,{0.2f,0.4f,0.8f},"surface attenuation color");
    maps.materials[0].roughness = 0;
    near(surfaceAt(maps,{{0.25f,0.25f,1},{0,0,-1}},generatedHit).roughness,0,"geometry does not impose shader roughness floor");

    Scene interpolated = singleTriangle();
    interpolated.triangles[0].v[0].normal = {0,0,1};
    interpolated.triangles[0].v[1].normal = normalize(Vec3{1,0,1});
    interpolated.triangles[0].v[2].normal = normalize(Vec3{0,1,1});
    for (Vertex& vertex : interpolated.triangles[0].v) vertex.tangent = {1,0,0,1};
    interpolated.finalize();
    const Surface smooth = surfaceAt(interpolated,{{0.25f,0.25f,1},{0,0,-1}},generatedHit);
    near(smooth.normal,normalize(Vec3{0,0,0.5f}+normalize(Vec3{1,0,1})*0.25f+normalize(Vec3{0,1,1})*0.25f),"interpolated shading normals");
    checkFrame(smooth,1,"interpolated orthogonal tangent frame");
    Scene parallel = singleTriangle(triangle({0,0,0},{1,-1,0},{1,0,-1}));
    for (Vertex& vertex : parallel.triangles[0].v) {
        vertex.normal = {1,1,1}; vertex.tangent = {1,1,1,1};
    }
    parallel.finalize();
    checkFrame(surfaceAt(parallel,{{1,1,1},{-1,-1,-1}},{1,0.25f,0.25f,0}),1,"parallel oblique tangent falls back to UV frame");
}

// BVH oracle: inspect every triangle, using the same primitive/material tests but
// no acceleration structure. Tie-breaking is explicit and independent of leaf order.
bool brute(const Scene& scene, const Ray& ray, Hit& hit, bool any = false) {
    bool found = false;
    const float upper = hit.t;
    for (uint32_t i = 0; i < scene.triangles.size(); ++i) {
        Hit candidate;
        candidate.t = upper; candidate.triangle = i;
        if (!intersectTriangle(scene.triangles[i],ray,candidate) || !scene.accepts(scene.triangles[i],ray,candidate)) continue;
        if (any) { hit = candidate; return true; }
        if (!found || candidate.t < hit.t || (candidate.t == hit.t && i < hit.triangle)) { found = true; hit = candidate; }
    }
    return found;
}
// Compare hit existence, identity, distance, barycentrics, and shadow visibility.
void compare(const Scene& scene, const Bvh& bvh, const Ray& ray, float upper, const std::string& label) {
    Hit reference{upper,0.123f,0.234f,999999}, accelerated = reference;
    const bool expected = brute(scene,ray,reference);
    require(bvh.intersect(scene,ray,accelerated) == expected,label+" closest-hit parity");
    if (expected) {
        near(accelerated.t,reference.t,label+" nearest distance");
        require(accelerated.triangle == reference.triangle,label+" nearest triangle: BVH="+std::to_string(accelerated.triangle)+", brute="+std::to_string(reference.triangle));
        near(accelerated.u,reference.u,label+" barycentric u"); near(accelerated.v,reference.v,label+" barycentric v");
    } else {
        require(accelerated.t == upper && accelerated.triangle == 999999,label+" no-hit output preserved");
    }
    Hit anyHit{upper}, bruteAny{upper};
    require(bvh.intersect(scene,ray,anyHit,true) == brute(scene,ray,bruteAny,true),label+" any-hit parity");
    if (expected) require(scene.accepts(scene.triangles[anyHit.triangle],ray,anyHit),label+" visibility uses alpha and culling");
}

// A cutout in front of a second surface must reveal it to both camera and shadow
// rays. Transmissive exits and double-sided rules are checked independently.
void testAlpha() {
    Scene scene;
    scene.materials.resize(2);
    scene.materials[0].alphaMode = AlphaMode::Mask;
    scene.materials[0].baseColorTexture.texture = addTexture(scene,{2,1,{{1,1,1,0},{1,1,1,1}}});
    scene.triangles.push_back(triangle({0,0,1},{1,0,1},{0,1,1},0));
    scene.triangles.push_back(triangle({0,0,0},{1,0,0},{0,1,0},1));
    scene.finalize();
    Bvh bvh; bvh.build(scene);
    const Ray throughHole{{0.2f,0.1f,2},{0,0,-1}}, covered{{0.7f,0.1f,2},{0,0,-1}};
    Hit holeHit, coverHit;
    require(bvh.intersect(scene,throughHole,holeHit) && holeHit.triangle == 1,"masked closest hit continues to geometry behind");
    require(bvh.intersect(scene,covered,coverHit) && coverHit.triangle == 0,"opaque mask coverage accepts front triangle");
    Hit visibility;
    require(!bvh.intersect(scene,{throughHole.origin,throughHole.direction,0,1.5f},visibility,true),"alpha-masked shadow hole");
    require(bvh.intersect(scene,{covered.origin,covered.direction,0,1.5f},visibility,true),"alpha-masked shadow blocker");
    scene.materials[0].baseColor.w = 0.5f;
    Hit equal;
    require(bvh.intersect(scene,covered,equal) && equal.triangle == 0,"alpha exactly at cutoff is covered");
    scene.materials[0].baseColor.w = std::nextafter(0.5f,0.0f);
    Hit below;
    require(bvh.intersect(scene,covered,below) && below.triangle == 1,"base-color alpha factor participates in mask");
    scene.materials[0].alphaMode = AlphaMode::Opaque;
    Hit opaque;
    require(bvh.intersect(scene,throughHole,opaque) && opaque.triangle == 0,"OPAQUE ignores texture and factor alpha");
    const Ray back{{0.7f,0.1f,-1},{0,0,1}};
    Hit backHit;
    require(!bvh.intersect(scene,back,backHit),"single-sided opaque backface culling");
    scene.materials[0].transmission = 0.7f;
    require(bvh.intersect(scene,back,backHit) && backHit.triangle == 0,"single-sided transmission allows refractive exit");
    const Surface exit = surfaceAt(scene,back,backHit);
    require(!exit.frontFace,"refractive exit preserves backface flag");
    near(exit.geometricNormal,{0,0,1},"refractive exit does not flip geometric normal");
    scene.materials[0].alphaMode = AlphaMode::Mask;
    Hit maskedExit;
    require(!bvh.intersect(scene,back,maskedExit,true),"transmissive visibility still honors alpha masks");
    scene.materials[0].baseColor.w = 1; scene.materials[0].transmission = 0; scene.materials[0].doubleSided = true;
    Hit twoSided;
    require(bvh.intersect(scene,back,twoSided,true),"double-sided backface accepted for visibility");
    for (const Ray& ray : {throughHole,covered,back}) compare(scene,bvh,ray,infinity,"mask query parity");
}

// Preparation may repair missing frames or drop zero-area triangles, but must
// diagnose invalid references, non-finite fields, and unrepresentable geometry.
void testValidation() {
    Scene scene = singleTriangle();
    scene.triangles.push_back(triangle({0,0,0},{1,0,0},{2,0,0}));
    scene.triangles.push_back(triangle({0,0,0},{0,0,0},{0,0,0}));
    scene.finalize();
    require(scene.triangles.size() == 1,"degenerate triangles removed");
    near(scene.triangles[0].area,0.5f,"triangle area computed");
    near(scene.triangles[0].geometricNormal,{0,0,1},"geometric normal computed");
    bool diagnostic = false;
    for (const std::string& warning : scene.warnings) if (warning.find("Skipped 2 degenerate") != std::string::npos) diagnostic = true;
    require(diagnostic,"degenerate input receives aggregate diagnostic");
    for (const Vertex& vertex : scene.triangles[0].v) {
        near(vertex.normal,{0,0,1},"missing normals generated");
        near(Vec3{vertex.tangent.x,vertex.tangent.y,vertex.tangent.z},{1,0,0},"missing tangents generated");
    }
    scene.finalize();
    require(scene.triangles.size() == 1,"repeat finalization stable");
    Scene skinny = singleTriangle(triangle({0,0,0},{1,0,0},{1,1e-20f,0}));
    skinny.finalize();
    require(skinny.triangles.size() == 1 && skinny.triangles[0].area > 0,"thin nondegenerate triangles are not discarded by arbitrary epsilon");
    Scene tiny = singleTriangle(triangle({0,0,0},{1e-30f,0,0},{0,1e-30f,0}));
    tiny.finalize();
    require(tiny.triangles.empty() && !tiny.warnings.empty(),"unrepresentable areas skipped diagnostically");
    Scene fallback = singleTriangle();
    fallback.materials[0].normalTexture.texture = solidTexture(fallback,{0.5f,0.5f,1,1});
    for (Vertex& vertex : fallback.triangles[0].v) vertex.uv[0] = {};
    fallback.finalize();
    checkFrame(surfaceAt(fallback,{{0.25f,0.25f,1},{0,0,-1}},{1,0.25f,0.25f,0}),1,"degenerate UV fallback");
    diagnostic = false;
    for (const std::string& warning : fallback.warnings) if (warning.find("degenerate UVs") != std::string::npos) diagnostic = true;
    require(diagnostic,"degenerate normal-map UV warning");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int field = 0; field < 6; ++field) {
        Scene invalid = singleTriangle();
        if (field == 0) invalid.triangles[0].v[0].position.x = nan;
        if (field == 1) invalid.triangles[0].v[0].normal.z = infinity;
        if (field == 2) invalid.triangles[0].v[0].tangent.x = nan;
        if (field == 3) invalid.triangles[0].v[0].tangent.w = nan;
        if (field == 4) invalid.triangles[0].v[0].uv[0].y = infinity;
        if (field == 5) invalid.triangles[0].v[0].uv[1].x = nan;
        throws([&] { invalid.finalize(); },"non-finite vertex field "+std::to_string(field),"Triangle 0");
    }
    Scene badIndex = singleTriangle(); badIndex.triangles[0].material = 1;
    throws([&] { badIndex.finalize(); },"invalid material index","material index");
    Scene noMaterials = singleTriangle(); noMaterials.materials.clear();
    throws([&] { noMaterials.finalize(); },"missing materials","material index");
    Scene badMaterial = singleTriangle(); badMaterial.materials[0].emission.x = nan;
    throws([&] { badMaterial.finalize(); },"non-finite material","Material 0");
    Scene badImage; badImage.images.push_back({2,2,{{1,1,1,1}}});
    throws([&] { badImage.finalize(); },"bad image dimensions","Image 0");
    Scene badPixels; badPixels.images.push_back({1,1,{{1,1,nan,1}}});
    throws([&] { badPixels.finalize(); },"non-finite image","non-finite");
    Scene badTexture; badTexture.textures.push_back({0});
    throws([&] { badTexture.finalize(); },"invalid image reference","Texture 0");
    Scene badSlot = singleTriangle(); badSlot.materials[0].normalTexture.texture = 4;
    throws([&] { badSlot.finalize(); },"invalid material texture","texture index");
    Scene huge = singleTriangle(triangle({0,0,0},{1e30f,0,0},{0,1e30f,0}));
    throws([&] { huge.finalize(); },"unrepresentable area","area");
}

// Mix random orientations/materials/ray bounds, then compare against brute force.
// Special distributions afterward exercise termination, ties, depth, and precision.
void testBvh() {
    Scene scene;
    scene.materials.resize(4);
    scene.materials[0].doubleSided = true;
    scene.materials[1].doubleSided = true; scene.materials[1].alphaMode = AlphaMode::Mask;
    scene.materials[1].baseColorTexture.texture = addTexture(scene,{2,2,{{1,1,1,0},{1,1,1,1},{1,1,1,1},{1,1,1,0}}},true,Wrap::Repeat);
    scene.materials[1].baseColorTexture.offset = {-0.1f,0.2f};
    scene.materials[3].transmission = 0.8f;
    Rng random(0x75645321);
    const auto uniform = [&](float lo, float hi) { return lo+(hi-lo)*random.uniform(); };
    for (uint32_t i = 0; i < 450; ++i) {
        const Vec3 center{uniform(-10,10),uniform(-10,10),uniform(-10,10)};
        const Vec3 normal = normalize(Vec3{uniform(-1,1),uniform(-1,1),uniform(-1,1)});
        const Frame frame(normal);
        const float scale = uniform(0.1f,2);
        scene.triangles.push_back(triangle(center-frame.x*scale-frame.y*scale,center+frame.x*scale,center+frame.y*scale,i%4));
    }
    scene.finalize();
    Bvh bvh;
    const auto buildStart = std::chrono::steady_clock::now();
    bvh.build(scene);
    const auto buildEnd = std::chrono::steady_clock::now();
    require(bvh.nodeCount() > 1 && bvh.nodeCount() < 2*scene.triangles.size(),"nontrivial compact BVH");
    require(bvh.bytes() >= bvh.nodeCount()*sizeof(Bounds),"BVH memory reporting");
    for (int i = 0; i < 6500; ++i) {
        Vec3 origin{uniform(-15,15),uniform(-15,15),uniform(-15,15)};
        Vec3 direction = normalize(Vec3{uniform(-1,1),uniform(-1,1),uniform(-1,1)});
        if (i%2 == 0) {
            const Triangle& target = scene.triangles[static_cast<std::size_t>(i)%scene.triangles.size()];
            const Vec3 point = target.v[0].position*0.5f+target.v[1].position*0.2f+target.v[2].position*0.3f;
            direction = normalize(point-origin);
        }
        if (i%5 == 0) direction *= 3;
        const Ray ray{origin,direction,i%7 == 0 ? 0.5f : 0,i%3 == 0 ? uniform(1,30) : infinity};
        compare(scene,bvh,ray,i%11 == 0 ? 5 : infinity,"random ray "+std::to_string(i));
    }
    std::cout << "BVH: " << scene.triangles.size() << " triangles, " << bvh.nodeCount() << " nodes, " << bvh.bytes()
              << " allocated bytes, build " << std::chrono::duration<double,std::milli>(buildEnd-buildStart).count() << " ms; 6500 randomized rays\n";

    Scene repeated = singleTriangle();
    // Coincident centroids must form a valid leaf instead of endless subdivision.
    repeated.materials[0].doubleSided = true;
    repeated.triangles.assign(100,triangle());
    repeated.finalize(); bvh.build(repeated);
    require(bvh.nodeCount() == 1,"identical centroids terminate SAH subdivision");
    compare(repeated,bvh,{{0.25f,0.25f,1},{0,0,-1}},infinity,"overlapping triangle deterministic tie");
    for (int shift = -5; shift <= 5; ++shift) {
        Scene ties;
        ties.materials.emplace_back(); ties.materials[0].doubleSided = true;
        const float x = static_cast<float>(shift);
        const Triangle target = triangle({x,0,0},{x+1,0,0},{x,1,0});
        ties.triangles.assign(2,target);
        for (int i = 0; i < 60; ++i) {
            const float otherX = static_cast<float>((i*23)%61-30)*2;
            ties.triangles.push_back(triangle({otherX,0,-2},{otherX+1,0,-2},{otherX,1,-2}));
        }
        ties.finalize(); bvh.build(ties);
        compare(ties,bvh,{{x+0.25f,0.25f,1},{0,0,-25}},infinity,"rounded-distance reordered leaf tie "+std::to_string(shift));
    }
    Scene exponential;
    // Exponentially spaced geometry stresses unbalanced SAH candidates and stack depth.
    exponential.materials.emplace_back(); exponential.materials[0].doubleSided = true;
    for (int exponent = -80; exponent < 80; ++exponent) {
        const float x = std::ldexp(1.0f,exponent);
        exponential.triangles.push_back(triangle({x,0,0},{x,1,0},{x,0,1}));
    }
    exponential.finalize(); bvh.build(exponential);
    for (int exponent = -75; exponent < 75; exponent += 3) {
        const float x = std::ldexp(1.0f,exponent);
        compare(exponential,bvh,{{x,0.25f,0.25f},{-1,0,0}},infinity,"bounded-depth adversarial SAH "+std::to_string(exponent));
    }
    for (float scale : {1e-15f,1e-5f,1.0f,1e5f,1e15f}) {
        Scene boundary;
        boundary.materials.emplace_back(); boundary.materials[0].doubleSided=true;
        for (int i=0;i<20;++i) {
            const float x=static_cast<float>(i)*5*scale;
            boundary.triangles.push_back(triangle({x,0,0},{x+scale,0,0},{x,scale,0}));
        }
        boundary.finalize(); bvh.build(boundary);
        for (float speed : {1e-30f,1.0f,1e30f}) {
            for (Vec3 point : {Vec3{0,0,scale},Vec3{scale,0,scale},Vec3{0.5f*scale,0.5f*scale,scale}}) {
                compare(boundary,bvh,{point,{0,-0.0f,-speed}},infinity,"reciprocal slabs scaled vertex/edge");
                compare(boundary,bvh,{point,{0,0,-speed}},0,"reciprocal slabs strict zero interval");
            }
        }
    }
    Scene empty;
    empty.finalize(); bvh.build(empty);
    require(bvh.nodeCount() == 0,"empty BVH build");
    Hit noHit;
    require(!bvh.intersect(empty,{{0,0,1},{0,0,-1}},noHit),"empty BVH closest query");
    require(!bvh.intersect(empty,{{0,0,1},{0,0,-1}},noHit,true),"empty BVH visibility query");
    Scene closeSurfaces;
    // Excluding a departed primitive must still find an interface inside the
    // ordinary origin-bias distance, including for any-hit shadow queries.
    closeSurfaces.materials.emplace_back(); closeSurfaces.materials[0].doubleSided=true;
    closeSurfaces.triangles.push_back(triangle({19,1e-7f,-1},{21,1e-7f,-1},{20,1e-7f,1}));
    closeSurfaces.triangles.push_back(triangle({18,0,-2},{20,0,2},{22,0,-2}));
    closeSurfaces.finalize(); bvh.build(closeSurfaces);
    const Ray onSurface{{20,1e-7f,0},{0,-1,0}};
    Hit self, next;
    require(bvh.intersect(closeSurfaces,onSurface,self)&&self.triangle==0,"unoffset ray identifies its source triangle");
    require(bvh.intersect(closeSurfaces,onSurface,next,false,0)&&next.triangle==1&&next.t>0,"source exclusion preserves an adjacent interface inside the usual offset");
    Hit shadow;
    require(bvh.intersect(closeSurfaces,onSurface,shadow,true,0)&&shadow.triangle==1,"source exclusion also applies to shadow queries");
    Scene ordered=singleTriangle();
    // Float t would merge these distances; double hit ordering must choose nearer geometry.
    ordered.triangles.push_back(triangle({0,0,1e-6f},{1,0,1e-6f},{0,1,1e-6f}));
    ordered.finalize(); bvh.build(ordered);
    Hit preciseHit;
    require(bvh.intersect(ordered,{{0.25f,0.25f,100},{0,0,-1}},preciseHit)&&preciseHit.triangle==1,
            "sub-float-distance separation retains true closest-hit ordering");
    require(preciseHit.t<100&&preciseHit.t>99.999998,"hit distance retains double precision");
    Scene reconstruction=singleTriangle(); reconstruction.finalize();
    // Reconstruct high+low coordinates and verify both triangle and box tests use them.
    const Surface reconstructed=surfaceAt(reconstruction,{{0,0,1},{0,0,-1}},{1,1.0/3,1.0/3,0});
    require(std::abs(static_cast<double>(reconstructed.position.x)+reconstructed.positionLow.x-1.0/3)<1e-14,
            "two-part hit origin retains barycentric reconstruction precision");
    const Ray preciseRay{{0.25f,0.25f,1},{0,0,-1},0,infinity,{0,0,1e-9f}};
    Hit lowHit;
    require(intersectTriangle(reconstruction.triangles[0],preciseRay,lowHit)&&lowHit.t>1&&lowHit.t<1.000000002,
            "triangle intersection includes the low origin component");
    Bounds preciseBox; preciseBox.expand({0,0,0}); preciseBox.expand({1,1,1});
    float nearBox=0;
    require(!preciseBox.intersect({{1,0.5f,0.5f},{0,1,0},0,infinity,{1e-9f,0,0}},nearBox),
            "parallel slab checks include low origin components");
}
}

int main() {
    try {
        // Keep analytic failures ahead of randomized tests for easier diagnosis.
        testIntersections();
        testOffsetsAndCamera();
        testTextureSampling();
        testNormalsAndSurfaces();
        testAlpha();
        testValidation();
        testBvh();
        std::cout << "Geometry tests passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Geometry tests FAILED after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
