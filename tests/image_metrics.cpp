// Headless validation for linear HDR artifacts. Tonemapped PNG values would
// hide high dynamic range and are not suitable for cross-build radiance checks.
#include "scene.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
    try {
        if (argc<2||argc>3) throw std::runtime_error("Usage: image_metrics image.hdr [comparison.hdr]");
        const auto image=cr::io::loadImage(argv[1],true);
        // Reject invalid energy before reporting dimensions, range, and brightness.
        cr::Vec3 maximum{}, minimum{cr::infinity}; double mean=0;
        for (const auto& p:image.pixels) {
            const cr::Vec3 color{p.x,p.y,p.z};
            if (!cr::finite(color)||p.x<0||p.y<0||p.z<0) throw std::runtime_error("Non-finite/negative image pixel");
            maximum=cr::max(maximum,color); minimum=cr::min(minimum,color); mean+=cr::luminance(color);
        }
        std::cout<<std::setprecision(9)<<"{\"width\":"<<image.width<<",\"height\":"<<image.height
                 <<",\"finite_pixels\":"<<image.pixels.size()<<",\"mean_luminance\":"<<mean/static_cast<double>(image.pixels.size())
                 <<",\"min_rgb\":["<<minimum.x<<','<<minimum.y<<','<<minimum.z<<"],\"max_rgb\":["<<maximum.x<<','<<maximum.y<<','<<maximum.z<<']';
        if (argc==3) {
            // Normalize total absolute RGB error by reference energy. Report the
            // largest individual error too, since a global mean can hide an outlier.
            const auto other=cr::io::loadImage(argv[2],true);
            if (image.width!=other.width||image.height!=other.height) throw std::runtime_error("Comparison dimensions differ");
            double error=0, energy=0; float maximumError=0;
            for (std::size_t i=0;i<image.pixels.size();++i) {
                const auto a=image.pixels[i], b=other.pixels[i];
                const cr::Vec3 delta{std::abs(a.x-b.x),std::abs(a.y-b.y),std::abs(a.z-b.z)};
                error+=delta.x+delta.y+delta.z; energy+=a.x+a.y+a.z; maximumError=std::max(maximumError,cr::maxComponent(delta));
            }
            const double relative=error/std::max(1e-12,energy);
            std::cout<<",\"relative_l1\":"<<relative<<",\"max_absolute_error\":"<<maximumError;
            // A tolerance permits harmless compiler/architecture rounding; this
            // utility does not require cross-platform byte-identical HDR files.
            if (relative>0.0005) { std::cout<<"}\n"; throw std::runtime_error("Cross-build radiance difference exceeds 0.05% tolerance"); }
        }
        std::cout<<"}\n"; return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
