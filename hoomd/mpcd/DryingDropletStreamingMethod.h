// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreamingMethod.h
 * \brief Declaration of mpcd::DryingDropletStreamingMethod
 */

#ifndef MPCD_DRYING_DROPLET_STREAMING_METHOD_H_
#define MPCD_DRYING_DROPLET_STREAMING_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/Variant.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include "BoundaryCondition.h"
#include "ConfinedStreamingMethod.h"
#include "SphereGeometry.h"

namespace mpcd
{
//! MPCD DryingDropletStreamingMethod
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in drying droplet.
 *
 * The integration scheme is essentially Verlet with specular reflections.First the radius and Velocity of droplet is updated then
 * the particle is streamed forward over the time interval. If the particle moves outside the Geometry(spherical droplet), it is placed back
 * on the boundary and particle velocity is updated according to the boundary conditions.Streaming then continues and
 * place the particle inside droplet(sphereGeometry). And particle which went outside or collided with droplet interface was marked.
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
         * \param density is solvent number density inside the droplet
         * \param seed is Seed to random number Generator
         */
        DryingDropletStreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                    unsigned int cur_timestep,
                                    unsigned int period,
                                    int phase,
                                    std::shared_ptr<::Variant> R,
                                    mpcd::detail::boundary bc,
                                    Scalar density,
                                    unsigned int seed);

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);

    protected:
        std::shared_ptr<::Variant> m_R;                    //!< Radius of Sphere
        mpcd::detail::boundary m_bc;                       //!< Boundary condition
        Scalar m_density;                                  //!< Solvent density
        unsigned int m_seed;                               //!< Seed to evaporator pseudo-random number generator
        unsigned int m_Npick;                              //!< Number of particles picked for evaporation on this rank
        const unsigned int m_mask = 1 << 1;                //!< Mask for flags

        GPUArray<unsigned int> m_bounced_index;            //!< Indices of bounced particles
        GPUVector<unsigned int> m_picks;                   //!< Particles picked for evaporation on this rank
        GPUVector<mpcd::detail::pdata_element> m_removed;  //!< Hold output particles that are removed

        virtual void applyPicks();                         //!< Apply the picks

    private:
        std::vector<unsigned int> m_all_picks;             //!< All picked particles on all the ranks

        unsigned int calculateNumBounced();                //!< For calculating N_bounced and m_bounced_index
        //!< For Making a random pick of particles across all ranks
        void makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned int N_bounced_total);
    };

namespace detail
{
//! Export mpcd::DryingDropletStreamingMethod to python
void export_DryingDropletStreamingMethod(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
