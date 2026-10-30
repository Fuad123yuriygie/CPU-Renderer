// Focused regressions for medium boundaries: which side absorbs a shadow ray,
// nearly touching surfaces, precision replay, and lighting an enclosed receiver.
#include "render.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace cr;
namespace {
// Keep checks active in optimized builds and include useful failure context.
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Compare RGB channels separately so one incorrect attenuation channel cannot hide.
void near(Vec3 actual, Vec3 expected, float tolerance, const std::string& message) {
    require(finite(actual) && finite(expected), message+": non-finite result");
    for (int channel=0;channel<3;++channel) {
        if (std::abs(actual[channel]-expected[channel])>tolerance)
            throw std::runtime_error(message+", channel "+std::to_string(channel)+": "+
                                     std::to_string(actual[channel])+" vs "+std::to_string(expected[channel]));
    }
}
// Build simple oriented surfaces directly for tests, independent of the glTF adapter.
void quad(Scene& scene, Vec3 a, Vec3 b, Vec3 c, Vec3 d, uint32_t material=0) {
    Triangle t;
    t.material=material;
    const Vec3 normal=normalize(cross(b-a,c-a));
    t.v[0].position=a; t.v[1].position=b; t.v[2].position=c;
    for (auto& v:t.v) v.normal=normal;
    scene.triangles.push_back(t);
    t.v[1].position=c; t.v[2].position=d;
    scene.triangles.push_back(t);
}
// A very wide but fully closed slab keeps central optical paths predictable
// while still satisfying the renderer's closed-volume contract.
Scene slab() {
    Scene scene;
    Material glass;
    glass.metallic=0; glass.roughness=0.6f; glass.transmission=1; glass.ior=1.5f;
    glass.thickness=1; glass.attenuationDistance=1; glass.attenuationColor={0.5f,0.8f,1};
    scene.materials.push_back(glass);
    // Outward, closed, one-unit slab. The camera is outside at z=2.
    constexpr float h=100;
    quad(scene,{-h,-h,1},{h,-h,1},{h,h,1},{-h,h,1});
    quad(scene,{h,-h,0},{-h,-h,0},{-h,h,0},{h,h,0});
    quad(scene,{-h,-h,0},{-h,-h,1},{-h,h,1},{-h,h,0});
    quad(scene,{h,-h,1},{h,-h,0},{h,h,0},{h,h,1});
    quad(scene,{-h,h,1},{h,h,1},{h,h,0},{-h,h,0});
    quad(scene,{-h,-h,0},{h,-h,0},{h,-h,1},{-h,-h,1});
    return scene;
}
// A narrow one-pixel camera approximates a known central ray. Disable roulette
// to isolate scattering/absorption rather than add another stochastic estimator.
Vec3 pixel(Scene& scene, const Environment& environment, int samples, int bounces, bool nee=true) {
    scene.finalize();
    Bvh bvh; bvh.build(scene);
    Camera camera; camera.position={0,0,2}; camera.yfov=1e-6f;
    RenderOptions options;
    options.width=options.height=1; options.samples=samples; options.bounces=bounces;
    options.threads=1; options.seed=42; options.russianRoulette=false; options.nextEventEstimation=nee;
    const Vec3 result=render(scene,bvh,camera,environment,options).pixels[0];
    require(finite(result) && std::min({result.x,result.y,result.z})>=0,"finite nonnegative transport");
    return result;
}
// On entry, the hypothetical shadow segment is already INSIDE the medium,
// even though the incoming camera segment was in air.
void testInternalPointLight() {
    Scene scene=slab();
    PunctualLight light;
    light.position={0,0,0.5f}; light.intensity=0.01f;
    scene.lights.push_back(light);
    const Vec3 absorbing=pixel(scene,Environment{},64,1);
    scene.materials[0].attenuationColor=Vec3{1};
    const Vec3 clear=pixel(scene,Environment{},64,1);
    require(clear.x>0.1f,"internal point light must illuminate the entry interface");
    // Only NEE at the entry contributes; its shadow segment is 0.5 units inside.
    near(absorbing,clear*Vec3{std::sqrt(0.5f),std::sqrt(0.8f),1},2e-6f,
         "entry shadow ray must use the new volume's Beer attenuation");
}
// On exit, an infinite shadow segment is in AIR. Retaining the old medium would
// incorrectly absorb the exterior light over an infinite distance.
void testExteriorDirectionalLight() {
    Scene scene=slab();
    scene.materials.push_back(scene.materials[0]);
    scene.materials[0].roughness=0;
    scene.triangles[2].material=scene.triangles[3].material=1;
    PunctualLight light;
    light.type=LightType::Directional; light.direction={0,0,1}; light.intensity=0.01f;
    scene.lights.push_back(light);
    const Vec3 absorbing=pixel(scene,Environment{},2048,2);
    for (auto& material:scene.materials) material.attenuationColor=Vec3{1};
    const Vec3 clear=pixel(scene,Environment{},2048,2);
    require(clear.x>0.01f,"exterior directional light must illuminate the exit interface");
    // Smooth entry fixes the in-medium path to one unit. The exit shadow is in air.
    near(absorbing,clear*Vec3{0.5f,0.8f,1},2e-6f,
         "exit shadow ray must not absorb over its infinite exterior segment");
}
// Independent BSDF-only and MIS estimators should converge to the same colored
// result. This catches systematic boundary-side errors, not just finite output.
void testAbsorbingEnvironmentMis() {
    Scene scene=slab();
    const Environment environment(Vec3{1});
    const Vec3 plain=pixel(scene,environment,131072,12,false);
    const Vec3 mis=pixel(scene,environment,131072,12,true);
    require(plain.x>0.25f && plain.x<plain.y && plain.y<plain.z,"slab must have visible colored absorption");
    // The original pre-scatter shadow medium lost about 0.017 red / 0.028 green.
    near(mis,plain,0.008f,"absorbing slab NEE and BSDF-only estimators must agree");
    std::cout << "  BSDF (" << plain.x << ',' << plain.y << ',' << plain.z << ") NEE ("
              << mis.x << ',' << mis.y << ',' << mis.z << ")\n";
}
// An opaque tilted face sits only micrometers above a closed liquid boundary.
// A normal-sized float origin offset can step over the liquid entry, so this
// intentionally exercises sample replay with precise, split-coordinate origins.
void testTouchingOpaqueBoundary() {
    Scene scene;
    Material liquid; liquid.metallic=0; liquid.transmission=1; liquid.ior=1;
    liquid.thickness=1; liquid.attenuationDistance=1; liquid.attenuationColor={0.5f,0.8f,1};
    Material opaque; opaque.metallic=0; opaque.ior=1; opaque.baseColor={0.6f,0.6f,0.6f,1};
    scene.materials={liquid,opaque};
    quad(scene,{0,0,20},{43,0,20},{43,0,-20},{0,0,-20});
    quad(scene,{0,-2,-20},{43,-2,-20},{43,-2,20},{0,-2,20});
    quad(scene,{0,-2,20},{43,-2,20},{43,0,20},{0,0,20});
    quad(scene,{43,-2,-20},{0,-2,-20},{0,0,-20},{43,0,-20});
    quad(scene,{0,-2,-20},{0,-2,20},{0,0,20},{0,0,-20});
    quad(scene,{43,-2,20},{43,-2,-20},{43,0,-20},{43,0,20});
    quad(scene,{-1,-1,21},{44,-1,21},{44,-1,-21},{-1,-1,-21},1);
    Triangle touching;
    touching.material=1;
    touching.v[0].position={19,-1+2e-6f,-1};
    touching.v[1].position={21,1+2e-6f,-1};
    touching.v[2].position={20,2e-6f,1};
    scene.triangles.push_back(touching);
    scene.finalize(); Bvh bvh; bvh.build(scene);
    Camera camera; camera.position={22,1,0}; camera.yfov=1e-7f;
    camera.forward=normalize(Vec3{20,2e-6f,0}-camera.position);
    camera.right=normalize(cross(camera.forward,{0,1,0})); camera.up=cross(camera.right,camera.forward);
    RenderOptions options; options.width=options.height=2; options.samples=256; options.bounces=8; options.threads=2;
    const auto result=render(scene,bvh,camera,Environment(Vec3{1}),options);
    require(result.boundaryRetries>0,"nearly touching opaque/dielectric surfaces exercise precision retracing");
    for(Vec3 p:result.pixels) require(finite(p)&&p.x>=0&&p.y>=0&&p.z>=0,"retraced boundary radiance is finite/nonnegative");
    options.threads=1;
    // Replay is tied to the same sample key, never to which worker found it.
    const auto serial=render(scene,bvh,camera,Environment(Vec3{1}),options);
    require(result.boundaryRetries==serial.boundaryRetries,"precision retry count is scheduling independent");
    for(std::size_t i=0;i<result.pixels.size();++i) require(lengthSquared(result.pixels[i]-serial.pixels[i])==0,"retraced pixels are worker deterministic");
    camera.position={20,-0.5f,0}; camera.forward={0,1,0}; camera.right={1,0,0}; camera.up={0,0,1};
    // Precision recovery must not disguise genuinely unsupported scene setup.
    bool rejected=false;
    try { (void)render(scene,bvh,camera,Environment(Vec3{1}),options); }
    catch(const std::runtime_error& error) { rejected=std::string(error.what()).find("outside closed volumes")!=std::string::npos; }
    require(rejected,"precision fallback must not hide an unsupported camera inside a volume");
    std::cout<<"  Replayed "<<result.boundaryRetries<<" of "<<result.cameraSamples<<" near-interface samples without dropping them\n";
}
// With equal indices and no absorption, a unit sky behind the closed boundary
// still gives this diffuse receiver exactly its albedo. Skipping guaranteed-
// blocked straight shadow rays must leave the refracted BSDF paths unchanged.
void testEnclosedOpaqueReceiver() {
    Scene scene=slab();
    scene.materials[0].ior=1;
    scene.materials[0].attenuationColor=Vec3{1};
    Material receiver; receiver.metallic=0; receiver.ior=1; receiver.baseColor={0.4f,0.4f,0.4f,1};
    scene.materials.push_back(receiver);
    quad(scene,{-99,-99,0.5f},{99,-99,0.5f},{99,99,0.5f},{-99,99,0.5f},1);
    const Vec3 nee=pixel(scene,Environment(Vec3{1}),4096,6,true);
    const Vec3 bsdf=pixel(scene,Environment(Vec3{1}),4096,6,false);
    near(nee,Vec3{0.4f},0.00001f,"enclosed opaque receiver has correct constant-sky energy");
    near(nee,bsdf,0.00001f,"skipping blocked infinite NEE preserves BSDF transport");
}
}
int main() {
    try {
        // Each test isolates one boundary mechanism, then the executable reports
        // a single failing status to CTest if any invariant is violated.
        const auto start=std::chrono::steady_clock::now();
        testInternalPointLight(); std::cout << "PASS finite internal point-light attenuation\n";
        testExteriorDirectionalLight(); std::cout << "PASS exterior directional-light exit attenuation\n";
        testAbsorbingEnvironmentMis(); std::cout << "PASS colored absorbing slab NEE/BSDF agreement\n";
        testTouchingOpaqueBoundary(); std::cout << "PASS touching opaque/dielectric surfaces and invalid camera rejection\n";
        testEnclosedOpaqueReceiver(); std::cout << "PASS constant-sky energy at an enclosed opaque receiver\n";
        std::cout << "Transport edge tests: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << " seconds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
