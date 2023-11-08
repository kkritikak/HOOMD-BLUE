// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "RandomSubsetPicker.h"

/* 
 * This file contains definitions of some functions used in RandomSubsetPicker::operator()
 */
void mpcd::RandomSubsetPicker::storePicksIdx(GPUArray<unsigned int>& picks)
    {
    picks.resize(m_Npick);
    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        {
        ArrayHandle<unsigned int> d_picks(picks, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_picks_idx(this->m_picks_idx, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_flags_idx(this->m_flags_idx, access_location::device, access_mode::read);

        m_storepickIdx_tuner->begin();
        mpcd::gpu::store_picks_idx(d_picks_idx.data,
                                   d_picks.data,
                                   d_flags_idx.data,
                                   this->m_Npick,
                                   m_storepickIdx_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_storepickIdx_tuner->end();
        }
    else
    #endif  //ENABLE_CUDA
        {
        ArrayHandle<unsigned int> h_picks(picks, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_picks_idx(m_picks_idx, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_flags_idx(m_flags_idx, access_location::host, access_mode::read);
        
        for (unsigned int i=0; i < m_Npick; ++i)
            {
            h_picks.data[i] = h_flags_idx.data[h_picks_idx.data[i]];
            }
        }
    }

void mpcd::RandomSubsetPicker::makeAllPicks(unsigned int timestep, unsigned int N_try_pick, unsigned int N_total_all_ranks)
    {
    /* The Fisher-Yates shuffle algorithm is applied to randomly pick unique particles
     * out of the possible particles across all ranks. The result is stored in
     * \a m_all_picks.
     */
    assert(N_try_pick < N_total_all_ranks);

    // fill up vector which we will randomly shuffle
    m_all_picks.resize(N_total_all_ranks);
    std::iota(m_all_picks.begin(), m_all_picks.end(), 0);

    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::RandomSubsetPicker, m_seed, timestep);

    // random shuffle (fisher-yates) to get picks, seeded the same across all ranks
    auto begin = m_all_picks.begin();
    auto end = m_all_picks.end();
    size_t left = std::distance(begin,end);
    unsigned int N_choose = N_try_pick;
    while (N_choose-- && left > 1)
        {
        hoomd::UniformIntDistribution rand_shift(left-1);

        auto r = begin;
        std::advance(r, rand_shift(rng));
        std::swap(*begin, *r);
        ++begin;
        --left;
        }

    // size the vector down to the number picked
    m_all_picks.resize(N_try_pick);
    }