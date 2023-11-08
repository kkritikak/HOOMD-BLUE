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
 * \param d_picks Indexes of picked particles in \a d_bounced_idx
 * \param d_bounced_idx Compacted array of particle indexes marked as candidates for evaporation
 * \param N_pick Number of picks made
 *
 * Using one thread per particle, d_bounced array is modified such that 3(11) is stored 
 * for picked particles .
 * See kernel::create_bounced_idx and mpcd::gpu::compact_bounced_idx for details of how bounced
 * particles index are stored.
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
 * \param d_picks Indexes of picked particles in \a d_bounced_idx
 * \param d_bounced_idx Compacted array of particle indexes marked as candidates for evaporation
 * \param N_pick Number of picks made
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