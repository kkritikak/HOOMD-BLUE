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
#include "Variant.h"
#include "BoundaryCondition.h"

namespace mpcd
{

//! MPCD confined streaming method
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles in Spherical geometry.
 *
 * \tparam Geometry The confining geometry (SphereGeometry).
 *
 * The integration scheme is essentially Verlet with specular reflections. The particle is streamed forward over
 * the time interval. If it moves outside the Geometry, it is placed back on the boundary and its velocity is
 * updated according to the boundary conditions. Streaming then continues until the timestep is completed.
 *
 * To facilitate this, every Geometry must supply three methods:
 *  1. detectCollision(): Determines when and where a collision occurs. If one does, this method moves the
 *                        particle back, reflects its velocity, and gives the time still remaining to integrate.
 *  2. isOutside(): Determines whether a particles lies outside the Geometry.
 *  3. validateBox(): Checks whether the global simulation box is consistent with the streaming geometry.
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
         * \param geom Streaming geometry
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
        std::shared_ptr<::variant> m_R; //!Radius of Sphere
        const boundary m_bc;            //!boundary conditions
    };

/*!
 * \param timestep Current time to stream
 */

void DryingDropletStreamingMethod<mpcd::SphereGeometry>::stream(unsigned int timestep)
    {
    const double start_R = (*m_R)(timestep);
    const double end_R = (*m_R)(timestep + m_period);
    const double V = (end_R - start_R)/(m_period * m_mpcd_dt);
    if (V>0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }
    m_validate_geom = m_geom; //validating the geometry
    m_geom = std::make_shared<mpcd::SphereGeometry>::stream(end_R, V, m_bc);
    //stream using ConfinedStreamingMethod
    ConfinedStreamingMethod<mpcd::SphereGeometry>::stream(timestep);
    //delete marked particles
    ArrayHandle<unsigned char> h_bounced(m_bounced, access_location::host, access_mode::overwrite);
    
    }


namespace detail
{
//! Export mpcd::DryingDropletStreaming to python
/*!
 * \param m Python module to export to
 */

void export_DryingDropletStreamingMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = "ConfinedStreamingMethod" + Geometry::getName();
    py::class_<mpcd::DryingDropletStreaming<mpcd::SphereGeometry>, std::shared_ptr<DryingDropletStreaming<mpcd::SphereGeometry>>>
        (m, name.c_str(), py::base<mpcd::ConfinedStreamingMethod<mpcd::SphereGeometry>>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::variant>, boundary>())
        .def_property("Spheregeometry" ,&mpcd::DryingDropletStreaming<mpcd::SphereGeometry>::setGeometry);
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
