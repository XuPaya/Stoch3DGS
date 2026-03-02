// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <3dgrt/pipelineParameters.h>
#include <3dgrt/kernels/cuda/gaussianParticles.cuh>
// clang-format on

extern "C" {
__constant__ PipelineBackwardParameters params;
}

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return PipelineParameters::InstancePrimitive ? optixGetInstanceIndex() : (PipelineParameters::CustomPrimitive ? optixGetPrimitiveIndex() : static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri));
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const float3& rayOri,
    const float3& rayDir,
    const float tmin,
    const float tmax,
    const float3& rayRadiance, // only used for backward tracing
    const float& rayTransmittanceGrad,
    const uint32_t& rayParticleID,
    const float& rayHitDistance,
    const float3& rayRadianceGrad,
    uint32_t randomSeed,
    bool isBackward = false) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    r0 = RayHit::InvalidParticleId; // Initialize all payload registers
    r1 = __float_as_uint(RayHit::InfiniteDistance);
    r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = r10 = r11 = r12 = r13 = r14 = r15 =
        r16 = r17 = r18 = r19 = r20 = r21 = r22 = r23 = r24 = r25 = r26 = r27 =
        r28 = r29 = r30 = r31 = __float_as_uint(RayHit::InfiniteDistance);
    // r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = RayHit::InvalidParticleId;
    // r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = __float_as_int(RayHit::InfiniteDistance);
    if (isBackward) {
        r0 = rayParticleID; // r0 holds the particle ID
        r1 = __float_as_uint(rayHitDistance); // r1 holds the hit distance
        r2 = __float_as_uint(rayRadiance.x); // r2-r4 hold the radiance components
        r3 = __float_as_uint(rayRadiance.y);
        r4 = __float_as_uint(rayRadiance.z);
        r5 = __float_as_uint(rayTransmittanceGrad); // r5 holds the transmittance
        r6 = __float_as_uint(0.0f); // r6 stores the second mmt
        r7 = __float_as_uint(0.0f); // r7 stores the grad vis
        r8 = __float_as_uint(rayRadianceGrad.x); // r8-r10 hold the radiance gradient components
        r9 = __float_as_uint(rayRadianceGrad.y);
        r10 = __float_as_uint(rayRadianceGrad.z); // r10 holds the radiance gradient z component
        r11 = 0; // r11 for counter
        r30 = 1;
        r31 = randomSeed; // r31 holds the random seed for stochastic sampling
    } else {
        r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = RayHit::InvalidParticleId;
        r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = __float_as_int(RayHit::InfiniteDistance);
        r30 = 0; // This payload can be used to pass a boolean flag from raygen to anyhit.
        r31 = randomSeed; // r31 holds the random seed for stochastic sampling
    }

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
               r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);

    rayPayload[0].particleId  = r0;
    rayPayload[0].distance    = __uint_as_float(r1);
    rayPayload[1].particleId  = r2;
    rayPayload[1].distance    = __uint_as_float(r3);
    rayPayload[2].particleId  = r4;
    rayPayload[2].distance    = __uint_as_float(r5);
    rayPayload[3].particleId  = r6;
    rayPayload[3].distance    = __uint_as_float(r7);
    rayPayload[4].particleId  = r8;
    rayPayload[4].distance    = __uint_as_float(r9);
    rayPayload[5].particleId  = r10;
    rayPayload[5].distance    = __uint_as_float(r11);
    rayPayload[6].particleId  = r12;
    rayPayload[6].distance    = __uint_as_float(r13);
    rayPayload[7].particleId  = r14;
    rayPayload[7].distance    = __uint_as_float(r15);
    rayPayload[8].particleId  = r16;
    rayPayload[8].distance    = __uint_as_float(r17);
    rayPayload[9].particleId  = r18;
    rayPayload[9].distance    = __uint_as_float(r19);
    rayPayload[10].particleId = r20;
    rayPayload[10].distance   = __uint_as_float(r21);
    rayPayload[11].particleId = r22;
    rayPayload[11].distance   = __uint_as_float(r23);
    rayPayload[12].particleId = r24;
    rayPayload[12].distance   = __uint_as_float(r25);
    rayPayload[13].particleId = r26;
    rayPayload[13].distance   = __uint_as_float(r27);
    rayPayload[14].particleId = r28;
    rayPayload[14].distance   = __uint_as_float(r29);
    rayPayload[15].particleId = r30;
    rayPayload[15].distance   = __uint_as_float(r31);
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return;
    }

    const float3 rayOrigin    = params.rayWorldOrigin(idx);
    const float3 rayDirection = params.rayWorldDirection(idx);

    float rayMaxHitDistance          = params.rayHitDistance[idx.z][idx.y][idx.x][1];
    float rayHitDistanceGrad   = params.rayHitDistanceGrad[idx.z][idx.y][idx.x][0];

    constexpr float epsT = 1e-9;

    float2 minMaxT   = intersectAABB(params.aabb, rayOrigin, rayDirection);
    float startT = fmaxf(0.0f, minMaxT.x - epsT);
    const float endT = fminf(rayMaxHitDistance, minMaxT.y) + epsT;
    int spp_bwd = 16;

    float gradVis = 0.f;
    params.rayGradVis[idx.z][idx.y][idx.x][0] = 0.0f;

    float3 rayRadianceGrad = make_float3(params.rayRadianceGrad[idx.z][idx.y][idx.x][0] / spp_bwd / 8.0f, 
                                        params.rayRadianceGrad[idx.z][idx.y][idx.x][1] / spp_bwd / 8.0f,  
                                        params.rayRadianceGrad[idx.z][idx.y][idx.x][2] / spp_bwd / 8.0f);
    float rayTransmittanceGrad = params.rayDensityGrad[idx.z][idx.y][idx.x][0] / spp_bwd / 8.0f;

#pragma unroll
    for (int i = 0; i < spp_bwd; i++) {
        RayPayload rayPayload;
        trace(rayPayload, rayOrigin, rayDirection, startT + epsT, endT, 
                    make_float3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, make_float3(0.0, 0.0, 0.0), 
                    params.frameNumber + i, false);
#pragma unroll
        for (int spp = 0; spp < 8; spp++)  // spp_fwd samples
        {
            uint32_t randomSeed = params.frameNumber + spp + 1;

            const RayHit rayHit = rayPayload[spp];


            float3 rayRadiance     = make_float3(0.f);
            float rayTransmittance = 1.f;
            float rayHitDistance   = 0.f;
            if (rayHit.particleId != RayHit::InvalidParticleId) {
                const bool acceptedHit = processHitStochastic<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                    rayOrigin,
                    rayDirection,
                    rayHit.particleId,
                    params.particleDensity,
                    params.particleRadiance,
                    params.hitMinGaussianResponse,
                    params.alphaMinThreshold,
                    params.sphDegree,
                    &rayTransmittance,
                    &rayRadiance,
                    &rayHitDistance,
        #ifdef ENABLE_NORMALS
                    &rayNormal
        #else
                    nullptr
        #endif
                );
                if (!acceptedHit) {
                    continue; // Skip to the next iteration if the hit was not accepted
                }

                RayPayload bwdRayPayload;
                trace(bwdRayPayload, rayOrigin, rayDirection, startT + epsT, rayHitDistance + 1e-3f, 
                    rayRadiance, rayTransmittanceGrad, rayHit.particleId, rayHitDistance, rayRadianceGrad, 
                    randomSeed, true);
                float gradVisI = bwdRayPayload[3].distance;
                gradVis += gradVisI;
            }
        }
    }
    params.rayGradVis[idx.z][idx.y][idx.x][0] = gradVis;
}

extern "C" __global__ void __intersection__is() {
    float hitDistance;
    const bool intersect = PipelineParameters::InstancePrimitive ? intersectInstanceParticle(optixGetObjectRayOrigin(),
                                                                                             optixGetObjectRayDirection(),
                                                                                             optixGetInstanceIndex(),
                                                                                             optixGetRayTmin(),
                                                                                             optixGetRayTmax(),
                                                                                             params.hitMaxParticleSquaredDistance,
                                                                                             hitDistance)
                                                                 : intersectCustomParticle(optixGetWorldRayOrigin(),
                                                                                           optixGetWorldRayDirection(),
                                                                                           optixGetPrimitiveIndex(),
                                                                                           params.particleDensity,
                                                                                           optixGetRayTmin(),
                                                                                           optixGetRayTmax(),
                                                                                           params.hitMaxParticleSquaredDistance,
                                                                                           hitDistance);
    if (intersect) {
        optixReportIntersection(hitDistance, 0);
    }
}


extern "C" __global__ void __anyhit__ah() { // sorting removed
    const bool isBackward = optixGetPayload_30() != 0; // Retrieve the boolean flag from the payload
    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};
    float3 rayOrigin    = optixGetWorldRayOrigin();
    float3 rayDirection = optixGetWorldRayDirection();
    unsigned int randomSeed = optixGetPayload_31(); // Retrieve the random seed from the payload
    uint3 idx = optixGetLaunchIndex();
    unsigned int seed = ((idx.x * 73856093u) ^ (idx.y * 19349663u) ^ (idx.z * 83492791u) ^ hit.particleId ^ randomSeed * 2654435761u);
    if (!isBackward) {
        // float density = params.particleDensity[hit.particleId].density;
        float curr_distance;
        // Use a better seed to reduce correlation: combine launch index and primitive id
        // unsigned int seed = optixGetPayload_31();
        float rand1d;
        float density = getDensityStochastic<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                                                            rayOrigin, rayDirection, hit.particleId,
                                                            params.particleDensity);
    
    #define STOCHASTIC_HIT_PAYLOAD_FWD(i_id, i_distance)                                      \
        {                                                                                 \
            curr_distance = __uint_as_float(optixGetPayload_##i_distance());               \
            seed = 1664525u * seed + 1013904223u;                                         \
            rand1d = (seed & 0x00FFFFFF) / float(0x01000000);                       \
            if (hit.distance < curr_distance && rand1d < density && density > params.alphaMinThreshold) { \
                optixSetPayload_##i_id(hit.particleId);                                 \
                optixSetPayload_##i_distance(__float_as_uint(hit.distance));            \
            }                                                                             \
        }
        
        STOCHASTIC_HIT_PAYLOAD_FWD(0, 1)
        STOCHASTIC_HIT_PAYLOAD_FWD(2, 3)
        STOCHASTIC_HIT_PAYLOAD_FWD(4, 5)
        STOCHASTIC_HIT_PAYLOAD_FWD(6, 7)
        STOCHASTIC_HIT_PAYLOAD_FWD(8, 9)
        STOCHASTIC_HIT_PAYLOAD_FWD(10, 11)
        STOCHASTIC_HIT_PAYLOAD_FWD(12, 13)
        STOCHASTIC_HIT_PAYLOAD_FWD(14, 15)
        optixIgnoreIntersection();
    } else {  
        unsigned int finParticleID = optixGetPayload_0();
        float fin_distance = __uint_as_float(optixGetPayload_1());
        float3 fin_radiance = make_float3(__uint_as_float(optixGetPayload_2()),
                                                __uint_as_float(optixGetPayload_3()),
                                                __uint_as_float(optixGetPayload_4()));
        bool isHit = hit.particleId == finParticleID;
        
        float sec_mmt = __uint_as_float(optixGetPayload_6());
        float grad_vis = __uint_as_float(optixGetPayload_7());
        float rayTransmittanceGrad = __uint_as_float(optixGetPayload_5());
        float3 rayRadianceGrad = make_float3(__uint_as_float(optixGetPayload_8()),
                                                __uint_as_float(optixGetPayload_9()),
                                                __uint_as_float(optixGetPayload_10()));
        float grad_vis_iter = 0.f;
        processHitStochasticBwd<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
            rayOrigin,
            rayDirection,
            hit.particleId,
            fin_distance,
            params.particleDensity,
            params.particleDensityGrad,
            params.particleRadiance,
            params.particleRadianceGrad,
            &grad_vis_iter,
            params.hitMinGaussianResponse,
            params.alphaMinThreshold,
            params.minTransmittance,
            params.sphDegree,
            rayTransmittanceGrad,
            rayRadianceGrad,
            fin_radiance,
            isHit
        );
        optixSetPayload_6(__float_as_uint(sec_mmt));
        optixSetPayload_7(__float_as_uint(grad_vis + grad_vis_iter));
        optixIgnoreIntersection();
    }
}