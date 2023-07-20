// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreaming.h
 * \brief Declaration of mpcd::DryingDropletStreaming
 */

#ifndef MPCD_DRYING_DROPLET_STREAMING_H_
#define MPCD_DRYING_DROPLET_STREAMING_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ConfinedStreamingMethod.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

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
template<class Geometry>
class PYBIND11_EXPORT DryingDropletStreaming : public mpcd::ConfinedStreamingMethod
    {
    public:
        //! Constructor
        /*!
         * \param sysdata MPCD system data
         * \param cur_timestep Current system timestep
         * \param period Number of timesteps between collisions
         * \param phase Phase shift for periodic updates
         * \param geom Streaming geometry
         * \param R is the radius of sphere intially at very first timestep
         */
        DryingDropletStreaming(std::shared_ptr<mpcd::SystemData> sysdata,
                                unsigned int cur_timestep,
                                unsigned int period,
                                int phase,
                                std::shared_ptr<const Geometry> geom, Scalar R)
        : mpcd::ConfinedStreamingMethod(sysdata, cur_timestep, period, phase),
          m_geom(geom), m_validate_geom(true)
          {}

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);

        //! Get the streaming geometry
        std::shared_ptr<const Geometry> getGeometry() const
            {
            return m_geom;
            }

        //! Set the streaming geometry
        void setGeometry(std::shared_ptr<const Geometry> geom)
            {
            m_validate_geom = true;
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const Geometry> m_geom; //!< Streaming geometry
        bool m_validate_geom;   //!< If true, run a validation check on the geometry

        //! Validate the system with the streaming geometry
        void validate();

        //! Check that particles lie inside the geometry
        virtual bool validateParticles();
    };

/*!
 * \param timestep Current time to stream
 */
template<class Geometry>
void DryingDropletStreaming<Geometry>::stream(unsigned int timestep)
    {
    
    }

template<class Geometry>
void DryingDropletStreaming<Geometry>::validate()
    {
    // ensure that the global box is padded enough for periodic boundaries
    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar cell_width = m_mpcd_sys->getCellList()->getCellSize();
    if (!m_geom->validateBox(box, cell_width))
        {
        m_exec_conf->msg->error() << "ConfinedStreamingMethod: box too small for " << Geometry::getName() << " geometry. Increase box size." << std::endl;
        throw std::runtime_error("Simulation box too small for confined streaming method");
        }

    // check that no particles are out of bounds
    unsigned char error = !validateParticles();
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_UNSIGNED_CHAR, MPI_LOR, m_exec_conf->getMPICommunicator());
    #endif // ENABLE_MPI
    if (error)
        throw std::runtime_error("Invalid MPCD particle configuration for confined geometry");
    }

/*!
 * Checks each MPCD particle position to determine if it lies within the geometry. If any particle is
 * out of bounds, an error is raised.
 */
template<class Geometry>
bool DryingDropletStreaming<Geometry>::validateParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::read);

    for (unsigned int idx = 0; idx < m_mpcd_pdata->getN(); ++idx)
        {
        const Scalar4 postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (m_geom->isOutside(pos))
            {
            m_exec_conf->msg->errorAllRanks() << "MPCD particle with tag " << h_tag.data[idx] << " at (" << pos.x << "," << pos.y << "," << pos.z
                          << ") lies outside the " << Geometry::getName() << " geometry. Fix configuration." << std::endl;
            return false;
            }
        }

    return true;
    }

namespace detail
{
//! Export mpcd::DryingDropletStreaming to python
/*!
 * \param m Python module to export to
 */
template<class Geometry>
void export_DryingDropletStreaming(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = "ConfinedStreamingMethod" + Geometry::getName();
    py::class_<mpcd::DryingDropletStreaming<Geometry>, std::shared_ptr<mpcd::DryingDropletStreaming<Geometry>>>
        (m, name.c_str(), py::base<mpcd::ConfinedStreamingMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<const Geometry>>())
        .def_property("geometry", &mpcd::DryingDropletStreaming<Geometry>::getGeometry,&mpcd::DryingDropletStreaming<Geometry>::setGeometry);
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_Drying_Droplet_Streaming_H_
