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
//! Updates d_bounced according to picks made by m_picker
cudaError_t apply_picks(unsigned int *d_bounced,
                        const unsigned int *d_picks,
                        const unsigned int m_mask,
                        unsigned int N_pick,
                        unsigned int block_size);
}
}

#endif // MPCD_DRYING_DROPLET_STREAMING_METHOD_CUH_