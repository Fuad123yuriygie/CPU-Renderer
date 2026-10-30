#pragma once
#include "scene.h"

namespace cr {
// Surface-scattering interface. Both directions point AWAY from the surface:
// outgoing points toward the previous path vertex, incoming toward the next one.
// A BSDF combines reflected light (BRDF) and transmitted light (BTDF).

// Fraction reflected by a dielectric interface; also detects total internal reflection.
float fresnelDielectric(float cosine, float etaIncident, float etaTransmitted);
// Apply Snell's law. False means no transmitted direction exists at this angle.
bool refractDirection(Vec3 outgoing, Vec3 normal, float etaIncidentOverTransmitted, Vec3& incoming);
// Convert a color specified at a reference distance into per-channel extinction.
Vec3 absorptionCoefficient(Vec3 color, float distance);
// Fraction of radiance remaining after a segment through a non-scattering medium.
Vec3 beerLambert(Vec3 coefficient, float distance);
// Continuous BSDF value and its complete mixture density, measured per steradian.
struct BsdfEval { Vec3 value{}; float pdf=0; };
struct BsdfSample {
    // World direction and the already-divided Monte Carlo throughput multiplier.
    Vec3 direction{}, weight{};
    // Delta events store their sampled branch probability instead of a continuous
    // density. eta is the transmitted/incident IOR ratio for a volume crossing.
    float pdf=0, eta=1;
    // Delta events are handled separately by MIS; transmitted controls medium state.
    bool delta=false, transmitted=false;
    // An empty/zero-weight sample terminates this path rather than inventing light.
    explicit operator bool() const { return maxComponent(weight)>0 && finite(weight) && finite(direction); }
};
// Evaluate the complete continuous mixture for next-event estimation and MIS.
BsdfEval evaluateBsdf(const Surface&, Vec3 outgoing, Vec3 incoming);
// Choose one scattering event and return a weight suitable for multiplying throughput.
BsdfSample sampleBsdf(const Surface&, Vec3 outgoing, Rng&);
}
