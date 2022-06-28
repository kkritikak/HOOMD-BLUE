// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::RejectionVirtualParticleFillerGPU
 */

#include "RejectionVirtualParticleFillerGPU.cuh"
#include "StreamingGeometry.h"

namespace mpcd
{
namespace gpu
{

//! Template instantiation of slit geometry
template cudaError_t draw_virtual_particles<mpcd::detail::SlitGeometry>
    (const draw_virtual_particles_args_t& args, const mpcd::detail::SlitGeometry& geom);

//! Template instantiation of slit-pore geometry
template cudaError_t draw_virtual_particles<mpcd::detail::SlitPoreGeometry>
    (const draw_virtual_particles_args_t& args, const mpcd::detail::SlitPoreGeometry& geom);

//! Template instantiation of sphere geometry
template cudaError_t draw_virtual_particles<mpcd::detail::SphereGeometry>
    (const draw_virtual_particles_args_t& args, const mpcd::detail::SphereGeometry& geom);

} // end namespace gpu
} // end namespace mpcd
