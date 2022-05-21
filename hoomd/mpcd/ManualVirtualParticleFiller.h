// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/VirtualParticleFiller.h
 * \brief Definition of class for backfilling solid boundaries with virtual particles.
 */

#ifndef MPCD_MANUAL_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_MANUAL_VIRTUAL_PARTICLE_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"
#include "SystemData.h"
#include "hoomd/Variant.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to the MPCD particle data
/*!
 * Virtual particles are used to pad cells sliced by solid boundaries so that their viscosity does not get too low.
 * The VirtualParticleFiller base class defines an interface for adding these particles. The base VirtualParticleFiller
 * implements a fill() method, which handles the basic tasks of appending a certain number of virtual particles to the
 * particle data. Each deriving class must then implement two methods:
 *  1. computeNumFill(), which is the number of virtual particles to add.
 *  2. drawParticles(), which is the rule to determine where to put the particles.
 */
class PYBIND11_EXPORT ManualVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
        ManualVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                              Scalar density,
                              unsigned int type,
                              std::shared_ptr<::Variant> T,
                              unsigned int seed);

        virtual ~ManualVirtualParticleFiller() {}

        //! Fill up virtual particles
        void fill(unsigned int timestep);

    protected:
        unsigned int m_N_fill;      //!< Number of particles to fill locally
        unsigned int m_first_tag;   //!< First tag of locally held particles
        unsigned int m_first_idx;   //!< Particle index to start adding particles from

        //! Compute the total number of particles to fill
        virtual void computeNumFill() {}

        //! Draw particles within the fill volume
        virtual void drawParticles(unsigned int timestep) {}
    };

namespace detail
{
//! Export the VirtualParticleFiller to python
void export_ManualVirtualParticleFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_VIRTUAL_PARTICLE_FILLER_H_
