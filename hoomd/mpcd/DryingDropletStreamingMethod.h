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
 *
 * The integration scheme is essentially Verlet with specular reflections.First SphereGeometry radius and Velocity is updated then
 * the particle is streamed forward over the time interval. If it moves outside the Geometry, it is placed back
 * on the boundary and particle velocity is updated according to the boundary conditions.Streaming then continues and
 * place the particle inside SphereGeometry. And particle which went outside or collided with geometry was marked.
 * Right amount of marked particles are then evaporated.
 *
 * To facilitate this, ConfinedStreamingMethod must flag the particles which were bounced back from surface.
 *
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
         */
        DryingDropletStreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                    unsigned int cur_timestep,
                                    unsigned int period,
                                    int phase, std::shared_ptr<::Variant> R, boundary bc)
        : mpcd::ConfinedStreamingMethod<mpcd::SphereGeometry>(sysdata, cur_timestep, period, phase, std::shared_ptr<mpcd::SphereGeometry>()),
          m_R(R),m_bc(bc)
          {}

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);
    private:
        std::shared_ptr<::Variant> m_R; //!Radius of Sphere
        const boundary m_bc;            //!boundary conditions
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
    
    //Removing the right amount of particles
    ArrayHandle<unsigned char> h_bounced(m_bounced, access_location::host, access_mode::read);
    
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
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, boundary>())
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
