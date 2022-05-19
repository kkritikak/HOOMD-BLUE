// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/VirtualParticleFiller.cc
 * \brief Definition of mpcd::VirtualParticleFiller
 */

#include "ManualVirtualParticleFiller.h"

mpcd::ManualVirtualParticleFiller::ManualVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                                   Scalar density,
                                                   unsigned int type,
                                                   std::shared_ptr<::Variant> T,
                                                   unsigned int seed)
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_N_fill(0), m_first_tag(0)
    {}

void mpcd::ManualVirtualParticleFiller::fill(unsigned int timestep)
    {
    // update the fill volume
    computeNumFill();

    // in mpi, do a prefix scan on the tag offset in this range
    // then shift the first tag by the current number of particles, which ensures a compact tag array
    m_first_tag = computeFirstTag(&m_N_fill);

    // add the new virtual particles locally
    const unsigned int first_idx = m_mpcd_pdata->addVirtualParticles(m_N_fill);

    // draw the particles consistent with those tags
    drawParticles(timestep);

    m_mpcd_pdata->invalidateCellCache();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ManualVirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::ManualVirtualParticleFiller, std::shared_ptr<mpcd::ManualVirtualParticleFiller> >(m, "ManualVirtualParticleFiller")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, Scalar, unsigned int, std::shared_ptr<::Variant>, unsigned int>())
        ;
    }
