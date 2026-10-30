// Manual fixed-scene phase benchmark. Timing is informational, not a CTest
// threshold, because CPU speed and system load vary between machines.
#include "render.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace cr;
int main(int argc,char** argv) {
    try {
        if (argc!=3) throw std::runtime_error("Usage: renderer_benchmark scene.glb sky.hdr");
        Scene scene=io::loadGltf(argv[1]);
        if (scene.cameras.empty()) throw std::runtime_error("Benchmark scene needs a camera");
        Bvh bvh; bvh.build(scene);
        Environment sky(io::loadImage(argv[2],true),0,1);
        // Reuse the same rays/hits between phases so later timings do not include
        // camera generation or BVH traversal a second time.
        constexpr int count=400000;
        std::vector<Ray> rays; rays.reserve(count);
        std::vector<Hit> hits; hits.reserve(count);
        std::vector<Ray> hitRays; hitRays.reserve(count);
        std::vector<Surface> surfaces;
        // Observable results prevent benchmark work from being optimized away.
        double checksum=0;
        const auto measure=[&](const char* name, std::size_t calls, auto function) {
            // One outer timer avoids adding a clock query to every measured call.
            const auto start=std::chrono::steady_clock::now(); function();
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::cout << std::fixed << std::setprecision(3) << name << ": " << seconds << " s, "
                      << seconds*1e9/static_cast<double>(calls) << " ns/call, " << calls << " calls\n";
        };
        Rng rng(42);
        // Uniform raster positions make the ray workload reproducible for a scene.
        measure("camera",count,[&] {
            for (int i=0;i<count;++i) rays.push_back(scene.cameras[0].generate(rng.uniform()*320,rng.uniform()*180,320,180));
        });
        measure("closest-hit traversal",count,[&] {
            for (const Ray& ray:rays) { Hit hit; if (bvh.intersect(scene,ray,hit)) { hits.push_back(hit); hitRays.push_back(ray); checksum+=hit.t; } }
        });
        // Visibility queries may stop at the first accepted blocker; closest-hit
        // queries must keep searching for a nearer triangle.
        measure("occlusion traversal",count,[&] {
            for (const Ray& ray:rays) { Hit hit; checksum+=bvh.intersect(scene,ray,hit,true)?1:0; }
        });
        surfaces.reserve(hits.size());
        // Shade only genuine hits, pairing each hit with the ray that produced it.
        measure("surface shading/maps",hits.size(),[&] {
            for (std::size_t i=0;i<hits.size();++i) surfaces.push_back(surfaceAt(scene,hitRays[i],hits[i]));
        });
        measure("BSDF sampling",surfaces.size(),[&] {
            for (std::size_t i=0;i<surfaces.size();++i) { const auto s=sampleBsdf(surfaces[i],-hitRays[i].direction,rng); checksum+=s.pdf; }
        });
        // Time evaluation independently from random direction generation.
        measure("BSDF evaluation",surfaces.size(),[&] {
            for (std::size_t i=0;i<surfaces.size();++i) { const auto e=evaluateBsdf(surfaces[i],-hitRays[i].direction,surfaces[i].normal); checksum+=e.pdf; }
        });
        measure("HDR importance sample+lookup+PDF",count,[&] {
            for (int i=0;i<count;++i) { float pdf=0; const Vec3 direction=sky.sample(rng,pdf); checksum+=luminance(sky.radiance(direction))+sky.pdf(direction); }
        });
        // Counts/checksum help detect unintended workload changes between runs.
        std::cout << "triangles=" << scene.triangles.size() << " hits=" << hits.size() << " checksum=" << checksum << '\n';
        return std::isfinite(checksum)?0:1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
