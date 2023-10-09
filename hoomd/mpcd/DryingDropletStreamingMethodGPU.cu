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
 * \param d_bounced Flags identifying which particles to select (1 = select)
 * \param d_bounced_idx Array of particle indexes
 * \param N Number of particles
 *
 * Using one thread per particle, d_bounced is checked if it's a 1.
 * The \a d_bounced_idx array is filled up with the particle indexes so that compact_bounced_idx
 * can later select these particle indexes based on \a d_bounced.
 */
__global__ void create_bounced_idx(unsigned int *d_bounced_idx,
                                   unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    d_bounced_idx[idx] = idx;

    }
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
                            const unsigned int *d_bounced_idx,
                            unsigned int mask,
                            unsigned int N_pick)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_pick) return;

    const unsigned int pick = d_picks[idx];
    const unsigned int pidx = d_bounced_idx[pick];

    d_bounced[pidx] |= mask;
    }
} //end namespace kernel

/*!
 * \param d_bounced Flags identifying which particles to select (1 = select)
 * \param d_bounced_idx Array of particle indexes
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \sa kernel::create_bounced_idx
 */
cudaError_t create_bounced_idx(unsigned int *d_bounced_idx,
                               unsigned int N,
                               unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::create_bounced_idx);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::create_bounced_idx<<<N/run_block_size + 1, run_block_size>>>(d_bounced_idx,
                                                                         N);

    return cudaSuccess;
    }

/*!
 * \param d_bounced_idx Compacted array of particle indexes marked as candidates for evaporation
 * \param d_num_bounced Flag to store the total number of bounced particles
 * \param d_tmp_storage Temporary storage allocated on the device, NULL on first call
 * \param tmp_storage_bytes Number of bytes necessary for temporary storage, 0 on first call
 * \param d_bounced Flags identifying which particles to select (1 = select)
 * \param N Number of particles
 *
 * The CUB library is used to compact the particle indexes of the selected particles
 * into \a d_bounced_idx based on the flags set in \a d_bounced. The number of bounced
 * particles is also determined.
 *
 * \note This function must be called twice. On the first call, the temporary storage
 *       required is sized and stored in \a tmp_storage_bytes. Device memory must be
 *       allocated to \a d_tmp_storage, and then the function can be called again
 *       to apply the transformation.
 *
 * \note Per CUB user group, DeviceSelect is in-place safe, and so input and output
 *       do not require a double buffer.
 *
 * See kernel::create_bounced_idx for details of how particles indices are stored
 */
cudaError_t compact_bounced_idx(unsigned int *d_bounced_idx,
                                unsigned int *d_num_bounced,
                                void *d_tmp_storage,
                                size_t &tmp_storage_bytes,
                                unsigned int *d_bounced,
                                unsigned int N)
    {
    if (N == 0) return cudaSuccess;

    HOOMD_CUB::DeviceSelect::Flagged(d_tmp_storage, tmp_storage_bytes, d_bounced_idx, d_bounced, d_bounced_idx, d_num_bounced, N);

    return cudaSuccess;
    }

/*!
 * \param d_picks Indexes of picked particles in \a d_bounced_idx
 * \param d_bounced_idx Compacted array of particle indexes marked as candidates for evaporation
 * \param N_pick Number of picks made
 * \param block_size Number of threads per block
 *
 * \sa kernel::evaporate_apply_picks
 */
cudaError_t apply_picks(unsigned int *d_bounced,
                        const unsigned int *d_picks,
                        const unsigned int *d_bounced_idx,
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
                                                                     d_bounced_idx,
                                                                     mask,
                                                                     N_pick);

    return cudaSuccess;
    }
} // end namespace gpu
} // end namespace mpcd