// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::RejectionVirtualParticleFillerGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

namespace mpcd
{
namespace gpu
{

//! Fill virtual particles using rejection counting method for a given geometry (spherical)
template<class Geometry>
cudaError_t draw_virtual_particles(const BoxDim& box,
                                   const Scalar mass,
                                   const unsigned int type,
                                   const unsigned int N_virt_max,
                                   const Scalar kT,
                                   const unsigned int timestep,
                                   const unsigned int seed,
                                   const unsigned int block_size,
                                   const unsigned int first_tag);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_
