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

//! MPCD RandomSubsetPicker
/*!
 * This class implements Random picking of particles, it picks randomly - N_try_pick particles out of \a N_total and
 * store the indices of picked particles
 *
 * First it calculates N_total(all candidate particles which can be possible picks) from flags [0 1 0 1 1 ...] which has 1),
 * This is done by a function CalculateTotalNumbersToPickFrom(),
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

 * In the existing code, you can call this like:
 * m_picker(m_picks, m_Npick, m_bounced, N_try_pick, timestep);
 */

namespace mpcd
{

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
            : m_sysdef(sysdef),
              m_exec_conf(sysdef->getParticleData()->getExecConf()),
              m_seed(seed),
              m_num_flags(m_exec_conf), 
              m_flags_idx(m_exec_conf),
              m_picks_idx(m_exec_conf)
            {
            #ifdef ENABLE_CUDA
            //!< tuner for creating indices of particles
            m_idx_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_Randomsubsetpicker_create_idx_particles", this->m_exec_conf));
            //!< Tuner for storing pick Indices in picks
            m_storepickIdx_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_Randomsubsetpicker_store_pick_Idx", this->m_exec_conf));
            #endif
            }

        template<typename T>
        void operator()(GPUArray<unsigned int>& picks,
                        unsigned int& N_pick,
                        const GPUArray<T>& flags,
                        unsigned int N_try_pick,
                        unsigned int timestep);

    protected:
        std::shared_ptr<SystemDefinition> m_sysdef;
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
        unsigned int m_seed;                                           //!< seed to random number generator
        unsigned int m_Npick;                                          //!< number of particles picked

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_idx_tuner;              //!< Tuner for creating flags idx
        std::unique_ptr<Autotuner> m_storepickIdx_tuner;     //!< Tuner for storing pick idx
        #endif //ENABLE_CUDA

        GPUFlags<unsigned int> m_num_flags;                  //!< GPU Flags for the number of flags which has 1 //this is for ENABLE_CUDA code
        GPUArray<unsigned int> m_flags_idx;                  //!< For storing index of flags which are 1 

        //! For calculating total number of particles which has 1 in flags[0 1 0 1 1 0...] and storing their index in flags_idx
        template<typename T>
        unsigned int CalculateTotalNumbersToPickFrom(const GPUArray<T>& flags);

        //!< For Making a random pick of particles across all ranks
        void makeAllPicks(unsigned int timestep, unsigned int N_try_pick, unsigned int N_total_all_ranks);
        //!< For storing indices of picked particles
        void storePicksIdx(GPUArray<unsigned int>& picks);

    private:
        std::vector<unsigned int> m_all_picks;               //!< All picked particles on all the ranks
        //!< Indices of Particles picked on this rank in a /m_flags_idx array( This will not contain the original index of picked particles in flags, these are index of particles picked in m_flags_idx))
        GPUArray<unsigned int> m_picks_idx;
    };

/*!
 * \param picks       //!< indexes that are picked in the *original* flags array
 * \param N_pick      //!< number of particles picked
 * \param flags       //!< flags indicating which particles to pick from [ 0 1 0 1 1 ...] 
 * \param N_try_pick  //!< target number of particles to pick (if total number of 1's in flags will be less than N_try_pick, it will pick all the 1's, in that case N_pick would be less then N_try_pick)
 * \param N           //!< Total number of particles
 * \param seed        //!< Seed to random number Generator
 */

template<typename T>
void RandomSubsetPicker::operator()(GPUArray<unsigned int>& picks,
                                    unsigned int& N_pick,
                                    const GPUArray<T>& flags,
                                    unsigned int N_try_pick,
                                    unsigned int timestep)
    {
    /*
     * This class select whether to take GPU or CPU code path depending on m_exec_conf.
     * This operator calculates total number of 1's in flags(N_total) and store the indices of those 1's in m_flags_idx.
     * then it calculates total number of 1's on all the ranks(N_total_all_ranks).
     * If N_total_all_ranks < N_try_pick it picks all the particles 
     * else it does more complicated picking logic
     *
     *
     * In the end it stores the indexes of picked particles(indices of picked particles according to the *original* flags array.) in picks and 
     * number of particles picked to N_pick indexes that were picked in the *original* flags array.
     */

    /*
     * First calculating How many 1's (candidate particles for picking) are there in flags and storing their indices in m_flags_idx
     * This is done by CalculateTotalNumbersToPickFrom()
     */

    unsigned int N_total = CalculateTotalNumbersToPickFrom<T>(flags);
    unsigned int N_total_all_ranks = N_total;
    unsigned int N_before = 0;

    /* 
     * Calculating all the total number of 1's(candidate particles for picking) in flags on all the ranks 
     */
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE, &N_total_all_ranks, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Exscan(&N_total, &N_before, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    /*
     * If N_try_pick >= N_total_all_ranks - pick all particles
     */
    if (N_total_all_ranks <= N_try_pick)
        {
        //pick all the particles
        m_picks_idx.resize(N_total);
        ArrayHandle<unsigned int> h_picks_idx(m_picks_idx, access_location::host, access_mode::overwrite);
        std::iota(h_picks_idx.data, h_picks_idx.data + N_total, 0);
        m_Npick = N_total;
        }

    else
        {
        // do the pick logic
        makeAllPicks(timestep, N_try_pick, N_total_all_ranks);
        /*
         * Select the picks that lie on my rank, with reindexing to local mark indexes.
         * This is performed in a do loop to allow for resizing of the GPUVector.
         * After a short time, the loop will be ignored.
         */
        const unsigned int max_pick_idx = N_before + N_total;
        bool overflowed = false;
        do
            {
            m_Npick = 0;
            const unsigned int max_Npick = m_picks_idx.getNumElements();

                {
                ArrayHandle<unsigned int> h_picks_idx(m_picks_idx, access_location::host, access_mode::overwrite);
                for (unsigned int i=0; i < N_try_pick; ++i)
                    {
                    const unsigned int pick = m_all_picks[i];
                    if (pick >= N_before && pick < max_pick_idx)
                        {
                        if (m_Npick < max_Npick)
                            {
                            h_picks_idx.data[m_Npick] = pick - N_before;
                            }
                        ++m_Npick;
                        }
                    }
                }

            overflowed = (m_Npick > max_Npick);
            if (overflowed)
                {
                m_picks_idx.resize(m_Npick);
                }

            } while (overflowed);
        }
    //storing idx of picked particles in picks and updating N_pick
    storePicksIdx(picks);
    N_pick = m_Npick;
    } //closing RandomSusetPicker::operator()
} //namespace mpcd

template<typename T>
unsigned int mpcd::RandomSubsetPicker::CalculateTotalNumbersToPickFrom(const GPUArray<T>& flags)
    {
    unsigned int N_total = 0;  
    unsigned int N = flags.getNumElements();  
    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        {
        m_flags_idx.resize(N);
        m_num_flags.resetFlags(0);
        ArrayHandle<unsigned int> d_flags_idx(m_flags_idx, access_location::device, access_mode::overwrite);
        // get the compact array of indexes of total particles which has 1 in flags
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
    } //end calculateTotalNumberToPickFrom()

#endif //MPCD_RANDOM_SUBSET_PICKER_H