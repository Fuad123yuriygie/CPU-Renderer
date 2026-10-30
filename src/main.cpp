// Command-line front end: parse and validate options, load assets, invoke the
// reusable renderer, then encode its linear film. Scene content stays in glTF.
#include "render.h"
#include <charconv>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>
// Windows APIs provide memory reporting and background-priority hints; the
// rendering algorithms themselves remain portable.
// Prevent their min/max macros from colliding with the standard-library functions.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {
using namespace cr;
namespace fs=std::filesystem;
// Keep supported features, defaults, and deliberate limitations visible through
// the executable itself; no authoring tool is needed to read usage or render.
void help() {
    std::cout << R"(CPU Renderer - portable C++20 CPU path tracer

Usage: cpu_renderer --scene FILE.gltf|FILE.glb [options]
  --camera NAME                    Named perspective camera (default: first)
  --environment FILE.hdr           Separate linear HDR sky (default: black)
  --environment-rotation DEGREES   Y-axis rotation of the entire sky (default: 0)
  --environment-intensity VALUE    Nonnegative sky multiplier (default: 1)
  --output FILE.png                PNG plus same-stem linear .hdr (renders/render.png)
  --preset preview|final           preview: 960x540, 32 spp; final: 1920x1080, 512 spp
  --width N --height N             Override preset resolution, each 1..16384
  --samples N                      Override preset sample count, 1..1048576
  --bounces N                      Maximum scattering events, 1..64 (default: 8)
  --threads N                      Workers, 0=half the logical CPUs (default: 0)
  --seed N                         Unsigned 64-bit random seed (default: 42)
  --exposure EV                    PNG exposure in stops, -24..24 (default: 0)
  --help                          Show this help

Explicit settings override presets regardless of argument order. Image size is
limited to 64 megapixels. Each pixel has a fixed sample order on any worker count.
PNG uses a filmic curve and sRGB encoding; HDR retains unexposed linear radiance.

Assets are local static glTF 2.0 triangle meshes. Supported extensions:
KHR_texture_transform, KHR_lights_punctual, KHR_materials_emissive_strength,
KHR_materials_transmission, KHR_materials_ior, KHR_materials_volume attenuation.
Use OPAQUE or MASK, not alpha BLEND. Bake procedural materials before export.
Thickness=0 is a zero-thickness parallel sheet with straight-through transmission
and GGX reflection; positive thickness marks a closed, outward-oriented volume
with rough refraction and Beer-Lambert absorption by world-space path length.
Keep refractive volumes disjoint/non-nested and cameras outside them. Scattering
media, advanced caustics, animation and GPU rendering are intentionally excluded.
Do not put a sun both in the HDR environment and in a punctual directional light.
Blender is for authoring only; ordinary builds and renders require no network.
)";
}
// Parse the entire argument without locale dependence or partial acceptance.
// Instantiating this with uint64_t also rejects negative random seeds.
template<class T> T integer(const std::string& text, const std::string& name) {
    T value{};
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if (error!=std::errc{} || end!=text.data()+text.size()) throw std::runtime_error("Invalid integer for "+name+": "+text);
    return value;
}
// Floating arguments must be finite, so NaN/infinity cannot reach exposure or sky math.
float real(const std::string& text, const std::string& name) {
    float value{};
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if (error!=std::errc{} || end!=text.data()+text.size() || !std::isfinite(value)) throw std::runtime_error("Invalid finite number for "+name+": "+text);
    return value;
}
// Normalize ASCII extensions and Windows path comparisons safely for signed char.
std::string lower(std::string value) {
    for (char& c:value) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
// Detect aliases such as hard links, symlinks, relative paths, and Windows case
// differences. weakly_canonical also works for an output that does not exist yet.
bool samePath(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::equivalent(a,b,ec) && !ec) return true;
    const auto aa=fs::weakly_canonical(fs::absolute(a)), bb=fs::weakly_canonical(fs::absolute(b));
#ifdef _WIN32
    return lower(aa.generic_string())==lower(bb.generic_string());
#else
    return aa==bb;
#endif
}
// Preflight every output BEFORE the long render. In particular, the paired HDR
// must not overwrite the input environment, nor may PNG overwrite an input map.
void prepareOutput(const fs::path& path, const std::vector<fs::path>& resources) {
    for (const auto& input:resources) if (samePath(path,input)) throw std::runtime_error("Output would overwrite an input asset: "+path.string());
    if (fs::exists(path)&&!fs::is_regular_file(path)) throw std::runtime_error("Output is not a regular file: "+path.string());
    const fs::path parent=fs::absolute(path).parent_path();
    fs::create_directories(parent);
    // An actual short write catches directory permissions/disk errors without
    // truncating an existing render just to test whether its folder is writable.
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto probe=parent/(".cpu-renderer-write-test-"+std::to_string(stamp));
    if (fs::exists(probe)) throw std::runtime_error("Output write probe collision");
    {
        std::ofstream stream(probe,std::ios::binary);
        if (!stream) throw std::runtime_error("Output directory is not writable: "+parent.string());
        stream.put('0'); stream.close();
        if (!stream) { fs::remove(probe); throw std::runtime_error("Output write test failed: "+parent.string()); }
    }
    fs::remove(probe);
    if (fs::exists(path)) {
        // Opening for in/out tests this file's permissions without truncation.
        std::fstream stream(path,std::ios::binary|std::ios::in|std::ios::out);
        if (!stream) throw std::runtime_error("Existing output is not writable: "+path.string());
    }
}
// Report the operating system's peak resident working set when available.
// Other platforms keep the renderer portable and report zero for this optional metric.
double peakMemoryMiB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),&counters,sizeof(counters))) return static_cast<double>(counters.PeakWorkingSetSize)/(1024*1024);
#endif
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        using namespace cr;
        // Collect validated option/value pairs before applying defaults. Reject
        // duplicate keys rather than allowing argument order to hide a mistake.
        std::map<std::string,std::string> args;
        const std::vector<std::string> known{"--scene","--camera","--environment","--environment-rotation","--environment-intensity","--output","--preset","--width","--height","--samples","--bounces","--threads","--seed","--exposure"};
        for (int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if (key=="--help") { help(); return 0; }
            if (std::find(known.begin(),known.end(),key)==known.end()) throw std::runtime_error("Unknown option: "+key+" (use --help)");
            if (i+1==argc) throw std::runtime_error("Missing value for "+key);
            if (args.contains(key)) throw std::runtime_error("Duplicate option: "+key);
            args[key]=argv[++i];
        }
        const auto value=[&](const std::string& key,const std::string& fallback) { auto it=args.find(key); return it==args.end()?fallback:it->second; };
        // Apply the preset first, then explicit overrides. Thus --width wins
        // whether it appears before or after --preset on the command line.
        RenderOptions options;
        const std::string preset=value("--preset","preview");
        if (preset=="final") { options.width=1920; options.height=1080; options.samples=512; }
        else if (preset!="preview") throw std::runtime_error("Preset must be preview or final");
        if (args.contains("--width")) options.width=integer<int>(args.at("--width"),"--width");
        if (args.contains("--height")) options.height=integer<int>(args.at("--height"),"--height");
        if (args.contains("--samples")) options.samples=integer<int>(args.at("--samples"),"--samples");
        if (args.contains("--bounces")) options.bounces=integer<int>(args.at("--bounces"),"--bounces");
        if (args.contains("--threads")) options.threads=integer<int>(args.at("--threads"),"--threads");
        if (args.contains("--seed")) options.seed=integer<uint64_t>(args.at("--seed"),"--seed");
        options.validate();
        // Exposure affects only PNG encoding. Environment rotation/intensity
        // affect the illumination used by both the PNG and raw HDR outputs.
        const float exposure=real(value("--exposure","0"),"--exposure");
        const float rotation=real(value("--environment-rotation","0"),"--environment-rotation");
        const float intensity=real(value("--environment-intensity","1"),"--environment-intensity");
        if (exposure < -24 || exposure > 24) throw std::runtime_error("Exposure must be in [-24, 24]");
        if (intensity<0 || intensity>1e6f) throw std::runtime_error("Environment intensity must be in [0, 1000000]");
        if (!args.contains("--scene")) throw std::runtime_error("--scene is required; use --help for usage");
        const fs::path scenePath=args.at("--scene"), output=value("--output","renders/render.png");
        if (lower(output.extension().string())!=".png") throw std::runtime_error("--output must have a .png extension (a linear .hdr is also written)");
        fs::path hdr=output; hdr.replace_extension(".hdr");
#ifdef _WIN32
        // The CLI is background rendering work: even explicit high worker counts
        // should yield CPU time to normal-priority interactive applications.
        if (!SetPriorityClass(GetCurrentProcess(),BELOW_NORMAL_PRIORITY_CLASS))
            std::cerr << "Warning: could not lower render priority; use --threads to reserve CPU capacity\n";
#endif
        // The adapter resolves glTF buffers/images relative to the scene file
        // and returns renderer-owned data plus all input paths for protection.
        std::cout << std::fixed << std::setprecision(3) << "Loading " << scenePath.string() << std::endl;
        auto stage=std::chrono::steady_clock::now();
        Scene scene=io::loadGltf(scenePath);
        const double loadingSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-stage).count();
        for (const auto& warning:scene.warnings) std::cerr << "Warning: " << warning << '\n';
        // This is the principal owned scene storage, separate from OS peak memory.
        std::size_t sceneBytes=scene.triangles.size()*sizeof(Triangle)+scene.materials.size()*sizeof(Material);
        for (const auto& image:scene.images) sceneBytes+=image.pixels.size()*sizeof(Vec4);
        std::cout << "Loaded " << scene.triangles.size() << " triangles, " << scene.materials.size() << " materials, "
                  << scene.images.size() << " images in " << loadingSeconds << " s; engine scene data "
                  << static_cast<double>(sceneBytes)/(1024*1024) << " MiB" << std::endl;
        if (scene.cameras.empty()) throw std::runtime_error("Scene has no perspective camera; export a named camera from the authoring tool");
        // No scene-specific camera is synthesized: select a camera from the asset.
        // On a name mismatch, list the exported choices in the diagnostic.
        const Camera* camera=&scene.cameras.front();
        if (args.contains("--camera")) {
            camera=nullptr;
            for (const auto& c:scene.cameras) if (c.name==args.at("--camera")) { camera=&c; break; }
            if (!camera) {
                std::string available;
                for (const auto& c:scene.cameras) available+=" '"+c.name+"'";
                throw std::runtime_error("Camera not found: "+args.at("--camera")+"; available:"+available);
            }
        }
        Environment environment;
        // The environment is deliberately a separate linear HDR input. It is
        // added to protected resources just like external glTF textures/buffers.
        if (args.contains("--environment")) {
            const fs::path path=args.at("--environment");
            if (lower(path.extension().string())!=".hdr") throw std::runtime_error("Environment must be a linear Radiance .hdr image");
            std::cout << "Loading HDR environment " << path.string() << std::endl;
            environment=Environment(io::loadImage(path,true),rotation,intensity);
            scene.resources.push_back(path);
        }
        // Missing lighting is usually an export/configuration error; fail clearly
        // instead of spending time producing an unexpectedly black image.
        bool illuminated=environment.integral()>0;
        for (const auto& light:scene.lights) illuminated|=luminance(light.color)*light.intensity>0;
        for (const auto& material:scene.materials) illuminated|=maxComponent(material.emission)>0;
        if (!illuminated) throw std::runtime_error("Scene has no nonzero lighting; supply --environment, punctual lights, or emissive geometry");
        scene.resources.push_back(scenePath);
        prepareOutput(output,scene.resources); prepareOutput(hdr,scene.resources);
        // Build acceleration once, then share it read-only with all render workers.
        std::cout << "Building binned-SAH BVH..." << std::endl;
        stage=std::chrono::steady_clock::now();
        Bvh bvh; bvh.build(scene);
        const double bvhSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-stage).count();
        std::cout << "BVH: " << bvh.nodeCount() << " nodes, " << static_cast<double>(bvh.bytes())/(1024*1024)
                  << " MiB, " << bvhSeconds << " s" << std::endl;
        const int requestedWorkers=options.threads?options.threads:static_cast<int>(std::max(1u,std::thread::hardware_concurrency()/2));
        const int workers=std::min(requestedWorkers,((options.width+15)/16)*((options.height+15)/16));
        std::cout << "Rendering camera '" << camera->name << "': " << options.width << 'x' << options.height << ", "
                  << options.samples << " spp, " << options.bounces << " bounces, " << workers << " workers, seed " << options.seed << std::endl;
        // render() handles parallelism and invokes this serialized progress callback.
        const RenderResult result=render(scene,bvh,*camera,environment,options,[](uint64_t done,uint64_t total,double seconds) {
            std::cout << "Progress " << 100.0*static_cast<double>(done)/static_cast<double>(total) << "% (" << done << '/'
                      << total << " camera samples), " << seconds << " s" << std::endl;
        });
        // Summarize the untonemapped film before encoding so clipping in the
        // display image cannot hide unexpectedly high or low radiance values.
        double mean=0; float maximum=0;
        for (Vec3 p:result.pixels) { mean+=luminance(p); maximum=std::max(maximum,maxComponent(p)); }
        mean/=static_cast<double>(result.pixels.size());
        std::cout << "Writing filmic sRGB PNG and unexposed linear HDR..." << std::endl;
        // Both files come from the same CPU-computed film; HDR is not reconstructed
        // from the display PNG and therefore retains its linear dynamic range.
        io::writePng(output,options.width,options.height,result.pixels,exposure);
        io::writeHdr(hdr,options.width,options.height,result.pixels);
        std::cout << "Render time: " << result.seconds << " s; " << static_cast<double>(result.cameraSamples)/result.seconds
                  << " camera samples/s; peak working set: " << peakMemoryMiB() << " MiB\n"
                  << "Boundary precision retries: " << result.boundaryRetries << " (same samples retraced, none discarded)\n"
                  << "Finite radiance: " << result.pixels.size() << " pixels; mean luminance " << mean << "; maximum channel " << maximum << '\n'
                  << "PNG: " << fs::absolute(output).string() << "\nHDR: " << fs::absolute(hdr).string() << std::endl;
        return 0;
    } catch (const std::bad_alloc&) {
        // Distinguish memory exhaustion from other failures while keeping both
        // nonzero: callers must not treat a failed run as a complete image pair.
        std::cerr << "Error: memory allocation failed; reduce asset size or output dimensions\n"; return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
