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

#include <string>
#include "ConfinedStreamingMethod.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/Variant.h"
#include "BoundaryCondition.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include <algorithm>

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

class PYBIND11_EXPORT DryingDropletStreamingMethod : public mpcd::ConfinedStreamingMethod<mpcd::detail::SphereGeometry>
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
                                    int phase, std::shared_ptr<::Variant> R, const Scalar density, unsigned int seed, mpcd::detail::boundary bc)
        : mpcd::ConfinedStreamingMethod<mpcd::detail::SphereGeometry>(sysdata, cur_timestep, period, phase, std::shared_ptr<mpcd::detail::SphereGeometry>()),
          m_R(R), m_density(density), m_seed(seed), m_picks(m_exec_conf), m_bc(bc)
          {}

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);

    protected:
        std::shared_ptr<::Variant> m_R;      //!< Radius of Sphere
        const Scalar m_density;              //!< Solvent density
        unsigned int m_seed;                 //!< Seed to evaporator pseudo-random number generator
        const mpcd::detail::boundary m_bc;   //!< Boundary condition
        unsigned int m_Npick;                              //!< Number of particles picked for evaporation on this rank

        GPUArray<unsigned int> m_bounced_index;           //!< Indices of bounced particles
        GPUVector<unsigned int> m_picks;                   //!< Particles picked for evaporation on this rank

        virtual void applyPicks();                          //!< Apply the picks

    private:
        std::vector<unsigned int> m_all_picks;             //!< All picked particles on all the ranks

        //!< For Making a random pick of particles across all ranks
        void makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned int N_bounced_total);

    };

/*!
 * \param timestep Current time to stream
 */

void DryingDropletStreamingMethod::stream(unsigned int timestep)
    {
    /*
     * Stream will only happen at streaming period and
     * geometry will only be updated when stream will happen
     */
    if(timestep%m_period != 0) return;
    
    //compute final Radius and Velocity of surface
    const Scalar start_R = m_R->getValue(timestep);
    const Scalar end_R = m_R->getValue(timestep + m_period);
    const Scalar V = (end_R - start_R)/(m_mpcd_dt);

    if (V > 0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }

    /*
     * If the initial geometry was not set. Set the geometry and validate the geometry 
     * Because the interface is shrinking, it is sufficient to validate only the first time the geometry
     * is set.
     */

    if (!m_geom)
        {
        m_geom = std::make_shared<mpcd::detail::SphereGeometry>(start_R, V , m_bc ); 
        validate();
        m_validate_geom = false;
        }

    /*
     *Update the geometry radius to the size at the end of the streaming step.
     * This needs to be done every time.
     */

    m_geom = std::make_shared<mpcd::detail::SphereGeometry>(end_R, V, m_bc );
    
    //stream according to base class rules
    ConfinedStreamingMethod<mpcd::detail::SphereGeometry>::stream(timestep);


    //calculating number of particles to evaporate(such that solvent density remain constant)
    const unsigned int N_remove = round((4.*M_PI/3.)*end_R*end_R*end_R*m_density);
    const int N_evap = m_mpcd_pdata->getNGlobal() - N_remove;

    // get the compact array of indexes of bounced particles
    unsigned int N_bounced = 0;
        {
        const unsigned int N = m_mpcd_pdata->getN();
        ArrayHandle<unsigned char> h_bounced(m_bounced, access_location::host, access_mode::read);

        bool overflowed = false;
        do
            {
            if (overflowed)
                {
                GPUArray<unsigned int> bounced_index(N_bounced, m_exec_conf);
                m_bounced_index.swap(bounced_index);
                // resize m_bounced_index with swap idiom
                }
            ArrayHandle<unsigned int> h_bounced_index(m_bounced_index, access_location::host, access_mode::overwrite);

            const unsigned int N_bounced_max = m_bounced_index.getNumElements();
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
            } while (overflowed);
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
    

    if (N_bounced_total <= N_evap)
        {
        //pick all bounced particles and remove all of them
        m_picks.resize(N_bounced);
        ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::overwrite);
        std::iota(h_picks.data, h_picks.data + N_bounced, 0);
        m_Npick = N_bounced;
        //calculating density if it will get changed alot - through a warning
        Scalar currentdensity = (m_mpcd_pdata->getNGlobal() - N_bounced_total)/((4.*M_PI/3.)*end_R*end_R*end_R);
        Scalar deltaindensity = std::fabs(currentdensity - m_density);
        if (deltaindensity > 0.1)
            {
            m_exec_conf->msg->warning() << "Solvent density changed to - " << currentdensity << std::endl;
            }
        }
    
    else
        {
        // do the pick logic
        makeAllPicks(timestep, N_evap , N_bounced_total);
        
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
                ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::overwrite);
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
    }

void DryingDropletStreamingMethod::makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned int N_bounced_total)
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

void DryingDropletStreamingMethod::applyPicks()
    {
    ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_bounced_index(m_bounced_index, access_location::host, access_mode::read);
    ArrayHandle<unsigned char> h_bounced(m_bounced, access_location::host, access_mode::readwrite);
    for (unsigned int i=0; i < m_Npick; ++i)
        {
        const unsigned int pidx = h_bounced_index.data[h_picks.data[i]];
        h_bounced.data[pidx]= 0b11;
        }
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
    py::class_<mpcd::DryingDropletStreamingMethod, mpcd::ConfinedStreamingMethod<mpcd::detail::SphereGeometry>, std::shared_ptr<DryingDropletStreamingMethod>>(m, "DryingDropletStreamingMethod")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, Scalar, unsigned int, boundary>());
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
