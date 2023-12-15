// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file RandomSubsetPicker.cuh
 * \brief Declaration of kernel drivers for RandomSubsetPicker
 */

#ifndef MPCD_RANDOM_SUBSET_PICKER_CUH_
#define MPCD_RANDOM_SUBSET_PICKER_CUH_

#include <cuda_runtime.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace mpcd
{
namespace gpu
{
//! For storing indices in m_flags_idx
cudaError_t create_flags_idx(unsigned int *d_flags_idx,
                             unsigned int N,
                             unsigned int block_size);

//! Drives CUB device selection routines for flags which are set to *1*
template<typename T>
cudaError_t compact_flags_idx(unsigned int *d_flags_idx,
                              unsigned int *d_num_flags,
                              void *d_tmp_storage,
                              size_t &tmp_storage_bytes,
                              T *d_flags,
                              unsigned int N);

//! Stores *original* indices of picked particles in \a d_picks according to d_picks_idx which contains indices of picked particles in a /d_flags_idx
cudaError_t store_picks_idx(unsigned int *d_picks_idx,
                            unsigned int *d_picks,
                            unsigned int *d_flags_idx,
                            unsigned int N_pick,
                            unsigned int block_size);

} //namespace gpu
} //namespace mpcd

#endif //MPCD_RANDOM_SUBSET_PICKER_CUH
