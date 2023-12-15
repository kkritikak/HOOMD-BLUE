// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RandomSubsetPicker.h
 * \brief Declaration of mpcd::RandomSubsetPicker
 */
#ifndef MPCD_RANDOM_SUBSET_PICKER_H
#define MPCD_RANDOM_SUBSET_PICKER_H

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_CUDA
#include "hoomd/Autotuner.h"
#include "RandomSubsetPicker.cuh"
#endif //ENABLE_CUDA

#include "hoomd/GPUArray.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/SystemDefinition.h"
#include <algorithm> 
#include <numeric>

namespace mpcd
{

//! MPCD RandomSubsetPicker
/*!
 * This class enables random particle selection. It attempts to randomly pick a specified number, N_try_pick, of particles from
 * \a GPUArray flags( marked with flags where 1 indicates that a particle can be picked)
 *
 * First it calculates N_total(number of all candidate particles which can be possible picks) from flags[0 1 0 1 1 ...],
 * This is done by a function countAndCompactFlags(),
 * then it calculates N_total_all ranks(all candidates particles which can be picked on all ranks)
 * Then if /param N_try_pick( that is number of particles to pick) >= N_total_all_ranks - it picks all the particles
 * else it does more complicated picking
 * 
 * In the end picks hold the Indices of particles picked in original flags array and N_pick would be number of particles picked.
 *
 * To use this in the any code-
 * Add the picker to the header: 
 * RandomSubsetPicker m_picker;
 * Construct the picker with the Constructor of class
 * m_picker(sysdef, seed);
 * In the code where you wanna use it, call this like:
 * m_picker(picks, m_Npick, m_flags, N_try_pick, timestep, N);
 */
class RandomSubsetPicker
    {
    public:
        //! Constructor
        /*!
         * \param sysdef MPCD system data
         * \param seed is Seed to random number Generator
         */
        RandomSubsetPicker(std::shared_ptr<SystemDefinition> sysdef,
                           unsigned int seed)
            : m_sysdef(sysdef), m_exec_conf(m_sysdef->getParticleData()->getExecConf()), m_seed(seed), m_num_flags(m_exec_conf), 
              m_flags_idx(m_exec_conf), m_picks_idx(m_exec_conf)
            {
            #ifdef ENABLE_CUDA
            m_idx_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_randomsubsetpicker_create_idx_particles", m_exec_conf));
            m_storepickIdx_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_randomsubsetpicker_store_pick_idx", m_exec_conf));
            #endif //ENABLE_CUDA
            }

        template<typename T>
        void operator()(GPUArray<unsigned int>& picks,
                        unsigned int& N_pick,
                        const GPUArray<T>& flags,
                        unsigned int N_try_pick,
                        unsigned int timestep,
                        unsigned int N);

    private:
        std::shared_ptr<SystemDefinition> m_sysdef;
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
        unsigned int m_seed;                                           //!< seed to random number generator
        unsigned int m_Npick;                                          //!< number of particles picked

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_idx_tuner;              //!< Tuner for creating flags idx
        std::unique_ptr<Autotuner> m_storepickIdx_tuner;     //!< Tuner for storing pick idx
        #endif //ENABLE_CUDA

        GPUFlags<unsigned int> m_num_flags;                  //!< GPU Flags for the number of flags which has 1 
        GPUArray<unsigned int> m_flags_idx;                  //!< For storing index of flags which are 1 

        //! For calculating total number of particles which has 1 in flags[0 1 0 1 1 0...] and storing their index in flags_idx
        template<typename T>
        unsigned int countAndCompactFlags(const GPUArray<T>& flags, unsigned int N);

        //! For picking up the particles that lies on current rank
        void makePicks(unsigned int timestep, unsigned int N_try_pick, unsigned int N_before, unsigned int N_total_all_ranks, unsigned int N_total);
        //! For storing indices of picked particles in picks and number of picked particles in N_pick
        void assignPicks(GPUArray<unsigned int>& picks, unsigned int& N_pick);

        std::vector<unsigned int> m_all_picks; //!< All picked particles on all the ranks
        GPUArray<unsigned int> m_picks_idx; //!< Indices of particles picked on this rank in \a m_flags_idx array
    };

/*!
 * \param picks       //!< indexes that are picked in the *original* flags array
 * \param N_pick      //!< number of particles picked
 * \param flags       //!< flags indicating which particles to pick from [ 0 1 0 1 1 ...] 
 * \param N_try_pick  //!< target number of particles to pick (if total number of 1's in flags will be less than N_try_pick, it will pick all the 1's, in that case N_pick would be less then N_try_pick
 * \param seed        //!< Seed to random number Generator
 * \param timestep    //!< timestep at which this is called
 * \param N           //!< total number of particles 
 */
template<typename T>
void RandomSubsetPicker::operator()(GPUArray<unsigned int>& picks,
                                    unsigned int& N_pick,
                                    const GPUArray<T>& flags,
                                    unsigned int N_try_pick,
                                    unsigned int timestep,
                                    unsigned int N)
    {
    /*
     * It first calculates total number of 1's in flags(N_total) and store the indices of those 1's in m_flags_idx.
     * then it calculates total number of 1's on all the ranks(N_total_all_ranks).
     * If N_total_all_ranks < N_try_pick it picks all the particles 
     * else it does more complicated picking logic
     *
     *
     * In the end it stores the indexes of picked particles(indices of picked particles according to the *original* flags array.) in picks and 
     * number of particles picked to N_pick indexes that were picked in the *original* flags array.
     */

    /*
     * First calculating How many marked particles are there in flags(1 = marked) and storing their indexes in m_flags_idx
     * This is done by countAndCompactFlags()
     */
    unsigned int N_total = countAndCompactFlags<T>(flags, N);
    unsigned int N_total_all_ranks = N_total;
    unsigned int N_before = 0;

    // reduce / scan the number of particles that are marked on all ranks
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE, &N_total_all_ranks, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Exscan(&N_total, &N_before, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    // If N_try_pick >= N_total_all_ranks - pick all particles
    if (N_total_all_ranks <= N_try_pick)
        {
        if (m_picks_idx.getNumElements() < N_total)
            {
            GPUArray<unsigned int> picks_indx(N_total, m_exec_conf);
            m_picks_idx.swap(picks_indx);
            }
        ArrayHandle<unsigned int> h_picks_idx(m_picks_idx, access_location::host, access_mode::overwrite);
        std::iota(h_picks_idx.data, h_picks_idx.data + N_total, 0);
        m_Npick = N_total;
        }
    else
        {
        // do the pick logic
        makePicks(timestep, N_try_pick, N_before, N_total_all_ranks, N_total);
        }

    // storing indexes of picked particles according to the *original* flags array in picks and updating N_pick
    assignPicks(picks, N_pick);
    } // closing RandomSubsetPicker::operator()
} // namespace mpcd

template<typename T>
unsigned int mpcd::RandomSubsetPicker::countAndCompactFlags(const GPUArray<T>& flags, unsigned int N)
    {
    unsigned int N_total = 0;  

    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        {
        if (m_flags_idx.getNumElements() < N)
            {
            GPUArray<unsigned int> flags_idx(N, m_exec_conf);
            m_flags_idx.swap(flags_idx);
            }
        m_num_flags.resetFlags(0);
        ArrayHandle<unsigned int> d_flags_idx(m_flags_idx, access_location::device, access_mode::overwrite);
        // get the compact array of indexes of total particles which are marked in flags
        m_idx_tuner->begin();
        mpcd::gpu::create_flags_idx(d_flags_idx.data,
                                    N,
                                    m_idx_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_idx_tuner->end();
        // use cub device select to filter out the total particles
            {
            ArrayHandle<T> d_flags(flags, access_location::device, access_mode::read);
            void *d_tmp_storage = NULL;
            size_t tmp_storage_bytes = 0;
            mpcd::gpu::compact_flags_idx<T>(d_flags_idx.data,
                                            m_num_flags.getDeviceFlags(),
                                            d_tmp_storage,
                                            tmp_storage_bytes,
                                            d_flags.data,
                                            N);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(this->m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp_storage = (void *)d_alloc();
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            mpcd::gpu::compact_flags_idx<T>(d_flags_idx.data,
                                            m_num_flags.getDeviceFlags(),
                                            d_tmp_storage,
                                            tmp_storage_bytes,
                                            d_flags.data,
                                            N);
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        N_total = m_num_flags.readFlags();
        }
    else
    #endif //ENABLE_CUDA
        {
        ArrayHandle<T> h_flags(flags, access_location::host, access_mode::read);

        bool overflowed = false;
        do
            {
            if (overflowed)
                {
                GPUArray<unsigned int> flags_index(N_total, m_exec_conf);
                m_flags_idx.swap(flags_index);
                }
            ArrayHandle<unsigned int> h_flags_idx(m_flags_idx, access_location::host, access_mode::overwrite);

            const unsigned int N_total_max = m_flags_idx.getNumElements();
            N_total = 0;
            for (unsigned int idx=0; idx < N; ++idx)
                {
                if (h_flags.data[idx])
                    {
                    if (N_total < N_total_max)
                        h_flags_idx.data[N_total] = idx;
                    ++N_total;
                    }
                }
            overflowed = N_total > N_total_max;
            } while (overflowed);
        }
    return N_total;
    }

#endif //MPCD_RANDOM_SUBSET_PICKER_H
