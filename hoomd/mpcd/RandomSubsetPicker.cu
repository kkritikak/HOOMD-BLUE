// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file RandomSubsetPicker.cu
 * \brief Definition of kernel drivers and kernels for RandomSubsetPicker
 */

#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/device/device_select.cuh>
#else
#include "hoomd/extern/cub/cub/device/device_select.cuh"
#endif
#include "RandomSubsetPicker.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
/*!
 * \param d_flags_idx Array of particle indexes
 * \param N Number of particles
 *
 * Using one thread per particle,
 * The \a d_flags_idx array is filled up with the particle indexes so that compact_flags_idx
 * can later select these particle indexes based on \a d_flags.
 */
__global__ void create_flags_idx(unsigned int *d_flags_idx,
                                 unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    d_flags_idx[idx] = idx;
    }

/*!
 * \param d_picks_idx Indexes of picked particles in \a d_flags_idx
 * \param d_picks Indexes of picked particles in \a d_flags
 * \param d_flags_idx Compacted array of particle indexes which has *1* in flags
 * \param N_pick Number of picks made
 *
 * Using one thread per particle, d_picks is modified such that picks has now indices of picked particles the *original* flags array.
 * See kernel::create_flags_idx and mpcd::gpu::compact_flags_idx for details of how
 * particles index are stored.
 */
__global__ void store_picks_idx(unsigned int *d_picks_idx,
                                unsigned int *d_picks,
                                const unsigned int *d_flags_idx,
                                unsigned int N_pick)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_pick) return;

    const unsigned int pick = d_picks_idx[idx];
    const unsigned int pidx = d_flags_idx[pick];

    d_picks[idx] = pidx;

    }
} //end namespace kernel

/*!
 * \param d_flags_idx Array of particle flags indexes
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \sa kernel::create_flags_idx
 */
cudaError_t create_flags_idx(unsigned int *d_flags_idx,
                             unsigned int N,
                             unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::create_flags_idx);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::create_flags_idx<<<N/run_block_size + 1, run_block_size>>>(d_flags_idx,
                                                                       N);
    return cudaSuccess;
    }

/*!
 * \param d_flags_idx Compacted array of particle indexes which has *1* in flags
 * \param d_num_flags Flag to store the total number of particles which has *1* in flags
 * \param d_tmp_storage Temporary storage allocated on the device, NULL on first call
 * \param tmp_storage_bytes Number of bytes necessary for temporary storage, 0 on first call
 * \param d_flags Flags identifying which particles to select (1 = select)
 * \param N Number of particles
 *
 * The CUB library is used to compact the particle indexes of the selected particles
 * into \a d_flags_idx based on the flags set in \a d_flags. The number of flagged
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
 * See kernel::create_flags_idx for details of how particles indices are stored
 */
template<typename T>
cudaError_t compact_flags_idx(unsigned int *d_flags_idx,
                              unsigned int *d_num_flags,
                              void *d_tmp_storage,
                              size_t &tmp_storage_bytes,
                              T *d_flags,
                              unsigned int N)
    {
    if (N == 0) return cudaSuccess;

    HOOMD_CUB::DeviceSelect::Flagged(d_tmp_storage, tmp_storage_bytes, d_flags_idx, d_flags, d_flags_idx, d_num_flags, N);

    return cudaSuccess;
    }

/*!
 * \param d_picks_idx Indexes of picked particles in \a d_flags_idx
 * \param d_picks Indexes of picked particles in \a d_flags
 * \param d_flags_idx Compacted array of particle indexes which has *1* in flags
 * \param N_pick Number of picks made
 * \param block_size Number of threads per block
 *
 * \sa kernel::store_picks_idx
 */
cudaError_t store_picks_idx(unsigned int *d_picks_idx,
                            unsigned int *d_picks,
                            unsigned int *d_flags_idx,
                            unsigned int N_pick,
                            unsigned int block_size)
    {
    if (N_pick == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::store_picks_idx);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::store_picks_idx<<<N_pick/run_block_size+1, run_block_size>>>(d_picks_idx,
                                                                         d_picks,
                                                                         d_flags_idx,
                                                                         N_pick);

    return cudaSuccess;
    }

//! Template instantiation of unsigned int
template cudaError_t compact_flags_idx<unsigned int>
    (unsigned int *d_flags_idx,
     unsigned int *d_num_flags,
     void *d_tmp_storage,
     size_t &tmp_storage_bytes,
     unsigned int *d_flags,
     unsigned int N);

//! Template instantiation of unsigned char
template cudaError_t compact_flags_idx<unsigned char>
    (unsigned int *d_flags_idx,
     unsigned int *d_num_flags,
     void *d_tmp_storage,
     size_t &tmp_storage_bytes,
     unsigned char *d_flags,
     unsigned int N);

} //namespace gpu
} //namespace mpcd