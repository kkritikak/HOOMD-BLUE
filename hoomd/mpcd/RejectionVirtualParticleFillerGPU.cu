// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::RejectionVirtualParticleFillerGPU
 */

#include "RejectionVirtualParticleFillerGPU.cuh"
#include "ParticleDataUtilities.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
/*
 * Using one thread per particle. The thread index is translated into a particle tag
 * and local particle index. A random position is drawn and a random velocity is 
 * drawn consistent with the system temperature.
*/
__global__ void draw_virtual_particles(Scalar4 *d_tmp_pos,
                                       Scalar4 *d_tmp_vel,
                                       bool d_track_bounded_particles,
                                       const 
                                       const BoxDim& box,
                                       const Scalar3 lo,
                                       const Scalar3 hi,
                                       const Scalar mass,
                                       const unsigned int type,
                                       const unsigned int N_virt_max,
                                       const Scalar kT,
                                       const unsigned int timestep,
                                       const unsigned int seed,
                                       const unsigned int block_size,
                                       const unsigned int first_tag,
                                       const unsigned int filler_id)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_virt_max)
        return;
    
    // initialize random number generator for positions and velocity
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::RejectionFiller, seed, timestep, first_tag+idx, filler_id);
    d_tmp_pos[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                   hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                   hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng),
                                   __int_as_scalar(m_type));
    
    hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);
    d_tmp_vel[pidx] = make_scalar4(vel.x,
                                   vel.y,
                                   vel.z,
                                   __int_as_scalar(mpcd::detail::NO_CELL));
    
    // check if particle is in bounds or out of bounds. in bound->0; out of bound->1
    

    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd
