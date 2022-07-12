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
#include "ParticleDataUtilities.h"
#include "hoomd/CachedAllocator.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"


namespace mpcd
{
namespace gpu
{

//! Common arguments passed to all geometry filling kernels
struct draw_virtual_particles_args_t
    {
    //! Constructor
    draw_virtual_particles_args_t(Scalar4 *_d_tmp_pos,
                                  Scalar4 *_d_tmp_vel,
                                  bool *_d_track_bounded_particles,
                                  const Scalar3 _lo,
                                  const Scalar3 _hi,
                                  const unsigned int _first_tag,
                                  const Scalar _vel_factor,
                                  const unsigned int _filler_id,
                                  const unsigned int _type,
                                  const unsigned int _N_virt_max,
                                  const unsigned int _timestep,
                                  const unsigned int _seed,
                                  const unsigned int _block_size)
        : d_tmp_pos(_d_tmp_pos), d_tmp_vel(_d_tmp_vel), d_track_bounded_particles(_d_track_bounded_particles),
        lo(_lo), hi(_hi), first_tag(_first_tag), vel_factor(_vel_factor), filler_id(_filler_id), type(_type),
        N_virt_max(_N_virt_max), timestep(_timestep), seed(_seed), block_size(_block_size)
        { }

    Scalar4 *d_tmp_pos;
    Scalar4 *d_tmp_vel;
    bool *d_track_bounded_particles;
    const Scalar3 lo;
    const Scalar3 hi;
    const unsigned int first_tag;
    const Scalar vel_factor;
    const unsigned int filler_id;
    const unsigned int type;
    const unsigned int N_virt_max;
    const unsigned int timestep;
    const unsigned int seed;
    const unsigned int block_size;
    };

// Function declarations
template<class Geometry>
cudaError_t draw_virtual_particles(const draw_virtual_particles_args_t& args, const Geometry& geom);

cudaError_t compact_virtual_particle_indices(const bool *d_flags,
                            const unsigned int num_items,
                            unsigned int *d_out,
                            unsigned int *d_num_selected_out);

cudaError_t copy_virtual_particles(unsigned int *d_compact_indices,
                          Scalar4 *d_positions,
                          Scalar4 *d_velocities,
                          unsigned int *d_tags,
                          const Scalar4 *d_temporary_positions,
                          const Scalar4 *d_temporary_velocities,
                          const unsigned int first_idx,
                          const unsigned int first_tag,
                          const unsigned int n_virtual,
                          const unsigned int block_size);

#ifdef NVCC
namespace kernel
{

//! Kernel to draw virtual particles outside any given geometry
/*!
 * \param d_tmp_pos Temporary positions
 * \param d_tmp_vel Temporary velocities
 * \param d_track_bounded_particles Particle tracking - in/out of given geometry
 * \param lo Left extrema of the sim-box
 * \param hi Right extrema of the sim-box
 * \param first_tag First tag (rng argument)
 * \param vel_factor Scale factor for uniform normal velocities consistent with particle mass / temperature
 * \param filler_id Identifier for the filler (rng argument)
 * \param type Particle type for filling
 * \param N_virt_max Maximum no. of virtual particles that can exist
 * \param timestep Current timestep
 * \param seed User seed for RNG
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \b implementation
 * We assign one thread per particle to draw random particle positions within the box and velocities consistent
 * with system temperature. Along with this a boolean array tracks if the particles are in/out of bounds of the
 * given geometry.
 */
template<class Geometry>
__global__ void draw_virtual_particles(Scalar4 *d_tmp_pos,
                                       Scalar4 *d_tmp_vel,
                                       bool *d_track_bounded_particles,
                                       const Scalar3 lo,
                                       const Scalar3 hi,
                                       const unsigned int first_tag,
                                       const Scalar vel_factor,
                                       const unsigned int filler_id,
                                       const unsigned int type,
                                       const unsigned int N_virt_max,
                                       const unsigned int timestep,
                                       const unsigned int seed,
                                       const unsigned int block_size,
                                       const Geometry geom)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_virt_max)
        return;

    // initialize random number generator for positions and velocity
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::RejectionFiller, seed, timestep, first_tag+idx, filler_id);
    Scalar3 pos = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));
    d_tmp_pos[idx] = make_scalar4(pos.x,
                                  pos.y,
                                  pos.z,
                                  __int_as_scalar(type));

    // check if particle is inside/outside the confining geometry
    d_track_bounded_particles[idx] = geom.isOutside(pos);

    hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);
    d_tmp_vel[idx] = make_scalar4(vel.x,
                                  vel.y,
                                  vel.z,
                                  __int_as_scalar(mpcd::detail::NO_CELL));
    }

/*!
 * \b implementation
 * Using one thread per particle, we assign the particle position, velocity and tags using the compacted indices
 * array as an input.
 */
__global__ void copy_virtual_particles(unsigned int *d_compact_indices,
                              Scalar4 *d_positions,
                              Scalar4 *d_velocities,
                              unsigned int *d_tags,
                              const Scalar4 *d_temporary_positions,
                              const Scalar4 *d_temporary_velocities,
                              const unsigned int first_idx,
                              const unsigned int first_tag,
                              const unsigned int n_virtual,
                              const unsigned int block_size)
    {
    // one thread per virtual particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_virtual)
        return;

    // d_compact_indices holds accepted particle indices from the temporary arrays
    const unsigned int pidx = d_compact_indices[idx];
    const unsigned int real_idx = first_idx + pidx;
    d_positions[real_idx] = d_temporary_positions[pidx];
    d_velocities[real_idx] = d_temporary_velocities[pidx];
    d_tags[real_idx] = first_tag + idx;
    }

} // end namespace kernel

/*!
 * \param args Common arguments for all geometries
 * \param geom Confined geometry
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \sa mpcd::gpu::kernel::draw_virtual_particles
 */
template<class Geometry>
cudaError_t draw_virtual_particles(const draw_virtual_particles_args_t& args, const Geometry& geom)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::draw_virtual_particles<Geometry>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(args.block_size, max_block_size);
    dim3 grid(args.N_virt_max / run_block_size + 1);
    mpcd::gpu::kernel::draw_virtual_particles<Geometry><<<grid, run_block_size>>>(args.d_tmp_pos, args.d_tmp_vel,
    args.d_track_bounded_particles, args.lo, args.hi, args.first_tag, args.vel_factor, args.filler_id, args.type,
    args.N_virt_max, args.timestep, args.seed, args.block_size, geom);

    return cudaSuccess;
    }


cudaError_t compact_virtual_particle_indices(const bool *d_flags,
                            const unsigned int num_items,
                            unsigned int *d_out,
                            unsigned int *d_num_selected_out)
    {
    void* d_tmp = NULL;
    size_t tmp_bytes = 0;
    cub::CountingInputIterator<int> itr(0);
    // Determine storage requirements
    cub::DeviceSelect::Flagged(d_tmp, tmp_bytes, itr, d_flags, d_out, d_num_selected_out, num_items);
    // virtual particles to keep
    ScopedAllocation<unsigned char> d_tmp_alloc(this->m_exec_conf->getCachedAllocator(),
                                                (tmp_bytes > 0) ? tmp_bytes : 1);
    d_tmp = (void*)d_tmp_alloc();
    // Run selection
    cub::DeviceSelect::Flagged(d_tmp, tmp_bytes, itr, d_flags, d_out, d_num_selected_out, num_items);
    return cudaSuccess;
    }


cudaError_t copy_virtual_particles(unsigned int *d_compact_indices,
                          Scalar4 *d_positions,
                          Scalar4 *d_velocities,
                          unsigned int *d_tags,
                          const Scalar4 *d_temporary_positions,
                          const Scalar4 *d_temporary_velocities,
                          const unsigned int first_idx,
                          const unsigned int first_tag,
                          const unsigned int n_virtual,
                          const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::copy_virtual_particles);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(n_virtual / run_block_size + 1);
    mpcd::gpu::kernel::copy_virtual_particles<<<grid, run_block_size>>>(d_compact_indices, d_positions,
                                                                        d_velocities, d_tags,
                                                                        d_temporary_positions, d_temporary_velocities,
                                                                        first_idx, first_tag, n_virtual, block_size);

    return cudaSuccess;
    }


#endif // NVCC

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_
