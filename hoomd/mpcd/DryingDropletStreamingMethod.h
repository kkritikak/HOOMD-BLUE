// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreaming.h
 * \brief Declaration of mpcd::DryingDropletStreaming
 */

#ifndef MPCD_DRYING_DROPLET_STREAMING_METHOD_H_
#define MPCD_DRYING_DROPLET_STREAMING_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ConfinedStreamingMethod.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/Variant.h"
#include "BoundaryCondition.h"

namespace mpcd
{

//! MPCD DryingDropletStreamingMethod
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in moving Spherical geometry.
 *
 * The integration scheme is essentially Verlet with specular reflections.First SphereGeometry radius and Velocity is updated then
 * the particle is streamed forward over the time interval. If it moves outside the Geometry, it is placed back
 * on the boundary and particle velocity is updated according to the boundary conditions.Streaming then continues and
 * place the particle inside SphereGeometry. And particle which went outside or collided with geometry was marked.
 * Right amount of marked particles are then evaporated.
 *
 * To facilitate this, ConfinedStreamingMethod must flag the particles which were bounced back from surface.
 */

class PYBIND11_EXPORT DryingDropletStreamingMethod : public mpcd::ConfinedStreamingMethod<mpcd::SphereGeometry>
    {
    public:
        //! Constructor
        /*!
         * \param sysdata MPCD system data
         * \param cur_timestep Current system timestep
         * \param period Number of timesteps between collisions
         * \param phase Phase shift for periodic updates
         * \param R is the radius of sphere
         * \param bc is boundary conditions(slip or no-slip)
         */
        DryingDropletStreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                    unsigned int cur_timestep,
                                    unsigned int period,
                                    int phase, std::shared_ptr<::Variant> R, boundary bc, const Scalar rho, unsigned int seed)
        : mpcd::ConfinedStreamingMethod<mpcd::SphereGeometry>(sysdata, cur_timestep, period, phase, std::shared_ptr<mpcd::SphereGeometry>()),
          m_R(R), m_bc(bc), m_rho(rho), m_seed(seed)
          {}

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);
    private:
        std::shared_ptr<::Variant> m_R; //!Radius of Sphere
        const boundary m_bc;            //!boundary conditions
        const Scalar m_rho;             //!solvent density

        GPUArray<unsigned int> m_bounced_index;          //!GPUArray for storing the indices or bounced particles
    };

/*!
 * \param timestep Current time to stream
 */

void DryingDropletStreamingMethod<mpcd::SphereGeometry>::stream(unsigned int timestep)
    {
    //compute final Radius and Velocity of surface
    const Scalar start_R = (*m_R)(timestep);
    const Scalar end_R = (*m_R)(timestep + m_period);
    const Scalar V = (end_R - start_R)/(m_period * m_mpcd_dt);
    
    if (V > 0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }
    
    /*
     * Update the geometry, and validate if the initial geometry was not set.
     * Because the interface is shrinking, it is sufficient to validate only the first time the geometry
     * is set.
     */
    
    m_validate_geom = m_geom; 
    m_geom = std::make_shared<const mpcd::SphereGeometry>(end_R, V, m_bc);

    
    //stream according to base class rules
    ConfinedStreamingMethod<mpcd::SphereGeometry>::stream(timestep);
    
    //calculating number of particles to remove from solvent density
    const unsigned int N_remove = round((4.*M_PI/3.)*end_R*end_R*end_R*m_rho);
    const int N_evap = m_mpcd_pdata->getNGlobal() - N_remove;
    
    // get the compact array of indexes of bounced particles
    unsigned int N_bounced = 0;
        {
        const unsigned int N = m_pdata->getN();
        ArrayHandle<unsigned char> h_bounced(m_bounced, access_location::host, access_mode::read);
        bool overflowed = false;
        do
            {
            if (overflowed)
                {
                GPUArray<unsigned char> bounced_index(N_bounced, m_exec_conf);
                m_bounced_index.swap(bounced_index);
                // resize m_bounced_index with swap idiom
                }
            ArrayHandle<unsigned int> h_bounced_index(m_bounced_index, access_location::host, access_mode::overwrite);

            const unsigned int N_bounced_max = m_bounced_index.size();
            N_bounced = 0;
            for (unsigned int idx=0; idx < N; ++idx)
                {
                if (h_bounced.data[idx])
                    {
                    if (N_bounced < N_bounced_max)
                        h_bounced_index.data[N_bounced] = idx;
                    ++N_bounced;
                    }
                }
            overflowed = N_bounced > N_bounced_max;
            } while (overflowed)
        }

    // reduce / scan the number of particles that were bounced on all ranks
    unsigned int N_bounced_total = N_bounced;
    unsigned int N_before = 0;
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(&N_bounced, &N_bounced_total, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Exscan(&N_bounced, &N_before, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    // N_remove is the total number to remove (computed previously)
    if (N_remove < N_bounced_total)
        {
        // do the pick logic, otherwise we should remove all the marked ones
        N_keep_total = N_bounced_total - N_remove;
        unsigned int N_pick_total;
        if (N_keep_total < N_bounced_total)
            {
            N_pick_total = N_keep_total;
            //pick particles for N_keep
            }
        else
            {
            N_pick_total = N_bounced_total;
            //pick particles for N_bounced
            }
        makeAllpicks(timestep, N_pick_total , N_bounced_total);
        
        /*
         * Select the picks that lie on my rank, with reindexing to local mark indexes.
         * This is performed in a do loop to allow for resizing of the GPUVector.
         * After a short time, the loop will be ignored.
         */
        
        const unsigned int max_pick_idx = N_before + N_bounced;
        overflowed = false;
        do
            {
            m_Npick = 0;
            const unsigned int max_Npick = m_picks.size();

                {
                ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::overwrite);
                for (unsigned int i=0; i < N_pick_total; ++i)
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
    }

void DryingDropletStreamingMethod::makeAllPicks(unsigned int timestep, unsigned int N_pick_total, unsigned int N_mark_total)
    {
    assert(N_pick_total <= N_mark_total);

    // fill up vector which we will randomly shuffle
    m_all_picks.resize(N_mark_total);
    std::iota(m_all_picks.begin(), m_all_picks.end(), 0);

    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::DryingDropletStreamingMethod, m_seed, timestep);

    // random shuffle (fisher-yates) to get picks, seeded the same across all ranks
    auto begin = m_all_picks.begin();
    auto end = m_all_picks.end();
    size_t left = std::distance(begin,end);
    unsigned int N_choose = N_pick_total;
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
    m_all_picks.resize(N_pick_total);
    }

namespace detail
{
//! Export mpcd::DryingDropletStreamingMethod to python
/*!
 * \param m Python module to export to
 */

void export_DryingDropletStreamingMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::DryingDropletStreamingMethod,mpcd::ConfinedStreamingMethod<mpcd::SphereGeometry>, std::shared_ptr<DryingDropletStreamingMethod>>
        (m, "DryingDropletStreamingMethod")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, boundary, Scalar, unsigned int>())
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
