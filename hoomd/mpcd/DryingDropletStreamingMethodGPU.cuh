// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file DryingDropletStreamingMethodGPU.cuh
 * \brief Declaration of kernel drivers for DryingDropletStreamingMethodGPU
 */

#ifndef MPCD_DRYING_DROPLET_STREAMING_METHOD_CUH_
#define MPCD_DRYING_DROPLET_STREAMING_METHOD_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace mpcd
{
namespace gpu
{
//! For scanning m_bounced and storing indices in bounced_idx
cudaError_t create_bounced_idx(unsigned int *d_bounced_idx,
                               unsigned int N,
                               unsigned int block_size);

//! Drives CUB device selection routines for bounced particles
cudaError_t compact_bounced_idx(unsigned int *d_bounced_idx,
                                unsigned int *d_num_bounced,
                                void *d_tmp_storage,
                                size_t &tmp_storage_bytes,
                                unsigned int *d_bounced,
                                unsigned int N);

//! Updates d_bounced according to picks made on cpu
cudaError_t apply_picks(unsigned int *d_bounced,
                        const unsigned int *d_picks,
                        const unsigned int *d_bounced_idx,
                        const unsigned int m_mask,
                        unsigned int N_pick,
                        unsigned int block_size);
}
}

#endif // MPCD_DRYING_DROPLET_STREAMING_METHOD_CUH_