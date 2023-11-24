// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreamingMethodGPU.h
 * \brief Declaration of mpcd::DryingDropletStreamingMethodGPU
 */

#ifndef MPCD_DRYING_DROPLET_STREAMING_METHOD_GPU_H_
#define MPCD_DRYING_DROPLET_STREAMING_METHOD_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/Variant.h"

#include "BoundaryCondition.h"
#include "ConfinedStreamingMethodGPU.h"
#include "RandomSubsetPicker.h"
#include "SphereGeometry.h"

namespace mpcd
{
//! MPCD DryingDropletStreamingMethodGPU
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in drying droplet on GPU
 *
 * See DryingDropletStreamingMethod for details
 */

class PYBIND11_EXPORT DryingDropletStreamingMethodGPU : public mpcd::ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>
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
        DryingDropletStreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
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

        GPUVector<unsigned int> m_picks;                   //!< Indices of Particles picked for evaporation on this rank in \a m_bounced array
        GPUVector<mpcd::detail::pdata_element> m_removed;  //!< Hold output particles that are removed

        std::unique_ptr<Autotuner> m_apply_picks_tuner;    //!< Tuner for applying picks

    private:
        RandomSubsetPicker m_picker;                       //!< For randomly picking particles to evaporate from particles which are bounced
    };

namespace detail
{
//! Export mpcd::DryingDropletStreamingMethod to python
void export_DryingDropletStreamingMethodGPU(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_GPU_H_
