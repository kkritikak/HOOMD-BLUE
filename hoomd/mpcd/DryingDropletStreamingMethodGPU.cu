// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward
/*!
 * \file DryingDropletStreamingMethodGPU.cu
 * \brief Definition of kernel drivers and kernels for DryingDropletStreamingMethodGPU
 */

#include "DryingDropletStreamingMethodGPU.cuh"

#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include "hoomd/extern/cub/cub/cub.cuh"
#endif

namespace mpcd
{
namespace gpu
{
namespace kernel
{
/*!
 * \param d_bounced Flags identifying which particles are bounced (1=bounced)
 * \param d_picks Indexes of picked particles in \a d_bounced
 * \param mask Mask for setting an additional bit in d_bounced
 * \param N_pick Number of picks made
 *
 * Using one thread per particle, the particles which were picked are marked 
 * by setting an additional bit in d_bounced array(e.g., 3 (11) is stored in m_bounced)
 * See RandomSubsetPicker::operator() on how particles are picked and picked particles
 * indices are stored in \a d_picks
 */
__global__ void apply_picks(unsigned int *d_bounced,
                            const unsigned int *d_picks,
                            unsigned int mask,
                            unsigned int N_pick)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_pick) return;

    const unsigned int pick = d_picks[idx];

    d_bounced[pick] |= mask;
    }
} //end namespace kernel

/*!
 * \param d_bounced flags identifying which particles were bounced (1=bounced)
 * \param d_picks Indexes of picked particles in \a d_bounced
 * \param mask Mask for setting an additional bit in d_bounced
 * \param N_pick Number of picks made out of all bounced particles
 * \param block_size Number of threads per block
 *
 * \sa kernel::apply_picks
 */
cudaError_t apply_picks(unsigned int *d_bounced,
                        const unsigned int *d_picks,
                        const unsigned int mask,
                        unsigned int N_pick,
                        unsigned int block_size)
    {
    if (N_pick == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::apply_picks);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::apply_picks<<<N_pick/run_block_size+1, run_block_size>>>(d_bounced,
                                                                     d_picks,
                                                                     mask,
                                                                     N_pick);

    return cudaSuccess;
    }
} // end namespace gpu
} // end namespace mpcd
