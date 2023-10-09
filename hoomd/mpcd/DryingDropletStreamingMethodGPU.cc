// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreamingMethodGPU.cc
 * \brief Declaration of mpcd::DryingDropletStreamingMethodGPU
 */
#include "DryingDropletStreamingMethodGPU.h"
#include "DryingDropletStreamingMethodGPU.cuh"
#include <algorithm>

//! Constructor
/*!
 * \param sysdata MPCD system data
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param R is the radius of sphere
 * \param bc is boundary conditions(slip or no-slip)
 * \param density is solvent number density inside the droplet
 * \param seed is Seed to random number Generator
 */
mpcd::DryingDropletStreamingMethodGPU::DryingDropletStreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                                       unsigned int cur_timestep,
                                                                       unsigned int period,
                                                                       int phase,
                                                                       std::shared_ptr<::Variant> R,
                                                                       mpcd::detail::boundary bc,
                                                                       Scalar density,
                                                                       unsigned int seed)
    : mpcd::ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>(sysdata, cur_timestep, period, phase, std::shared_ptr<mpcd::detail::SphereGeometry>()),
      m_R(R), m_bc(bc), m_density(density), m_seed(seed),m_picks(this->m_exec_conf), m_num_bounced(this->m_exec_conf), m_bounced_idx(this->m_exec_conf), m_removed(this->m_exec_conf)
    {
    m_idx_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mark_idx_particles", this->m_exec_conf));
    m_pick_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "pick_particles", this->m_exec_conf));    
    }

/*!
 * \param timestep Current time to stream
 */
void mpcd::DryingDropletStreamingMethodGPU::stream(unsigned int timestep)
    {
    // use peekStream since shouldStream will be called by parent class
    if(!this->peekStream(timestep)) return;

    // compute final Radius and Velocity of surface
    const Scalar start_R = m_R->getValue(timestep);
    const Scalar end_R = m_R->getValue(timestep + m_period);
    const Scalar V = (end_R - start_R)/(m_mpcd_dt);
    // checks if V <= 0, since size of droplet must decrease
    if (V > 0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }

    /*
     * If the initial geometry was not set. Set the geometry and validate the geometry
     * Because the interface is shrinking, it is sufficient to validate only the first time the geometry
     * is set.
     */
    if (!this->m_geom)
        {
        this->m_geom = std::make_shared<mpcd::detail::SphereGeometry>(start_R, V , m_bc );
        this->validate();
        this->m_validate_geom = false;
        }

    /*
     *Update the geometry radius to the size at the end of the streaming step.
     * This needs to be done every time.
     */
    this->m_geom = std::make_shared<mpcd::detail::SphereGeometry>(end_R, V, m_bc );

    //stream according to base class rules
    ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>::stream(timestep);

    //calculating number of particles to evaporate(such that solvent density remain constant)
    const unsigned int N_end = std::round((4.*M_PI/3.)*end_R*end_R*end_R*m_density);
    const unsigned int N_global = this->m_mpcd_pdata->getNGlobal();
    const unsigned int N_evap = (N_end < N_global) ? N_global - N_end : 0;
    // get the compact array of indexes of bounced particles and total N_bounced
    unsigned int N_bounced = calculateNumBounced();
    unsigned int N_bounced_total = N_bounced;
    unsigned int N_before = 0;
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE, &N_bounced_total, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Exscan(&N_bounced, &N_before, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    /* if N_bounced_total <= N_evap, pick all the particles to delete
     * else pick N_evap particles out of N_bounced_total
     */
    if (N_bounced_total <= N_evap)
        {
        //pick all bounced particles and remove all of them
        m_picks.resize(N_bounced);
        ArrayHandle<unsigned int> h_picks(this->m_picks, access_location::host, access_mode::overwrite);
        std::iota(h_picks.data, h_picks.data + N_bounced, 0);
        m_Npick = N_bounced;
        //calculating density if it will get changed alot, print a warning
        Scalar V_end = (4.*M_PI/3.)*end_R*end_R*end_R;
        Scalar currentdensity = (this->m_mpcd_pdata->getNGlobal() - N_bounced_total)/V_end;
        if (std::fabs(currentdensity - m_density) > Scalar(0.1))
            {
            this->m_exec_conf->msg->warning() << "Solvent density changed to: " << currentdensity << std::endl;
            }
        }
    else
        {
        // do the pick logic
        makeAllPicks(timestep, N_evap, N_bounced_total);

        /*
         * Select the picks that lie on my rank, with reindexing to local mark indexes.
         * This is performed in a do loop to allow for resizing of the GPUVector.
         * After a short time, the loop will be ignored.
         */
        const unsigned int max_pick_idx = N_before + N_bounced;
        bool overflowed = false;
        do
            {
            m_Npick = 0;
            const unsigned int max_Npick = m_picks.getNumElements();

                {
                ArrayHandle<unsigned int> h_picks(this->m_picks, access_location::host, access_mode::overwrite);
                for (unsigned int i=0; i < N_evap; ++i)
                    {
                    const unsigned int pick = m_all_picks[i];
                    if (pick >= N_before && pick < max_pick_idx)
                        {
                        if (m_Npick < max_Npick)
                            {
                            h_picks.data[m_Npick] = pick - N_before;
                            }
                        ++m_Npick;
                        }
                    }
                }

            overflowed = (m_Npick > max_Npick);
            if (overflowed)
                {
                m_picks.resize(m_Npick);
                }

            } while (overflowed);
        }

    applyPicks();
    //finally removing the picked particles
    this->m_mpcd_pdata->removeParticlesGPU(this->m_removed,
                                           this->m_bounced,
                                           this->m_mask,
                                           timestep);
    }
void mpcd::DryingDropletStreamingMethodGPU::applyPicks()
    {
    ArrayHandle<unsigned int> d_picks(this->m_picks, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bounced_idx(this->m_bounced_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bounced(this->m_bounced, access_location::device, access_mode::readwrite);
    m_pick_tuner->begin();
    mpcd::gpu::apply_picks(d_bounced.data,
                           d_picks.data,
                           d_bounced_idx.data,
                           this->m_mask,
                           this->m_Npick,
                           m_pick_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_pick_tuner->end();
    }

unsigned int mpcd::DryingDropletStreamingMethodGPU::calculateNumBounced()
    {
    if (this->m_mpcd_pdata->getN() == 0)
        return 0;

    m_bounced_idx.resize(m_mpcd_pdata->getN());
    m_num_bounced.resetFlags(0);
    ArrayHandle<unsigned int> d_bounced(this->m_bounced, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bounced_idx(this->m_bounced_idx, access_location::device, access_mode::overwrite);
    // get the compact array of indexes of bounced particles and total N_bounced

    m_idx_tuner->begin();
    mpcd::gpu::create_bounced_idx(d_bounced_idx.data,
                                  this->m_mpcd_pdata->getN(),
                                  m_idx_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_idx_tuner->end();
    // use cub device select to filter out the bounced particles
        {
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        mpcd::gpu::compact_bounced_idx(d_bounced_idx.data,
                                       m_num_bounced.getDeviceFlags(),
                                       d_tmp_storage,
                                       tmp_storage_bytes,
                                       d_bounced.data,
                                       m_mpcd_pdata->getN());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        ScopedAllocation<unsigned char> d_alloc(this->m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void *)d_alloc();
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        mpcd::gpu::compact_bounced_idx(d_bounced_idx.data,
                                       m_num_bounced.getDeviceFlags(),
                                       d_tmp_storage,
                                       tmp_storage_bytes,
                                       d_bounced.data,
                                       m_mpcd_pdata->getN());
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    const unsigned int N_bounced = m_num_bounced.readFlags();
    return N_bounced;
    }

void mpcd::DryingDropletStreamingMethodGPU::makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned int N_bounced_total)
    {
    assert(N_pick <= N_bounced_total);

    // fill up vector which we will randomly shuffle
    m_all_picks.resize(N_bounced_total);
    std::iota(m_all_picks.begin(), m_all_picks.end(), 0);

    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::DryingDropletStreamingMethod, m_seed, timestep);

    // random shuffle (fisher-yates) to get picks, seeded the same across all ranks
    auto begin = m_all_picks.begin();
    auto end = m_all_picks.end();
    size_t left = std::distance(begin,end);
    unsigned int N_choose = N_pick;
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
    m_all_picks.resize(N_pick);
    }
void mpcd::detail::export_DryingDropletStreamingMethodGPU(pybind11::module& m)
    {
    //! Export mpcd::DryingDropletStreamingMethod to python
    /*!
     * \param m Python module to export to
     */
    namespace py = pybind11;
    py::class_<mpcd::DryingDropletStreamingMethodGPU, mpcd::ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>, std::shared_ptr<mpcd::DryingDropletStreamingMethodGPU>>(m, "DryingDropletStreamingMethodGPU")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, boundary, Scalar, unsigned int>());
    }
