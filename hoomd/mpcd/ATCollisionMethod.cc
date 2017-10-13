// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethod.h
 * \brief Definition of mpcd::ATCollisionMethod
 */

#include "ATCollisionMethod.h"
#include "RandomNumbers.h"
#include "hoomd/Saru.h"

mpcd::ATCollisionMethod::ATCollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                           unsigned int cur_timestep,
                                           unsigned int period,
                                           int phase,
                                           unsigned int seed,
                                           std::shared_ptr<mpcd::CellThermoCompute> thermo,
                                           std::shared_ptr<mpcd::CellThermoCompute> rand_thermo,
                                           std::shared_ptr<::Variant> T)
    : mpcd::CollisionMethod(sysdata,cur_timestep,period,phase,seed),
      m_thermo(thermo), m_rand_thermo(rand_thermo), m_T(T)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD AT collision method" << std::endl;

    m_thermo->getCallbackSignal().connect<mpcd::ATCollisionMethod, &mpcd::ATCollisionMethod::drawVelocities>(this);
    }


mpcd::ATCollisionMethod::~ATCollisionMethod()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD AT collision method" << std::endl;

    m_thermo->getCallbackSignal().disconnect<mpcd::ATCollisionMethod, &mpcd::ATCollisionMethod::drawVelocities>(this);
    }

void mpcd::ATCollisionMethod::rule(unsigned int timestep)
    {
    m_thermo->compute(timestep);

    if (m_prof) m_prof->push("MPCD collide");
    // compute the cell average of the random velocities
    m_pdata->swapVelocities(); m_mpcd_pdata->swapVelocities();
    m_rand_thermo->compute(timestep);
    m_pdata->swapVelocities(); m_mpcd_pdata->swapVelocities();

    if (m_prof) m_prof->push(m_exec_conf, "apply");
    // apply random velocities
    applyVelocities();
    if (m_prof) m_prof->pop(m_exec_conf);

    if (m_prof) m_prof->pop();
    }

void mpcd::ATCollisionMethod::drawVelocities(unsigned int timestep)
    {
    // mpcd particle data
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getAltVelocities(), access_location::host, access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // embedded particle data
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_idx;
    std::unique_ptr< ArrayHandle<Scalar4> > h_vel_embed;
    std::unique_ptr< ArrayHandle<unsigned int> > h_tag_embed;
    if (m_embed_group)
        {
        h_embed_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(), access_location::host, access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getAltVelocities(), access_location::host, access_mode::overwrite));
        h_tag_embed.reset(new ArrayHandle<unsigned int>(m_pdata->getTags(), access_location::host, access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    // random velocities are drawn for each particle and stored into the "alternate" arrays
    const Scalar T = m_T->getValue(timestep);
    for (unsigned int idx=0; idx < N_tot; ++idx)
        {
        unsigned int pidx;
        unsigned int tag; Scalar mass;
        if (idx < N_mpcd)
            {
            pidx = idx;
            mass = m_mpcd_pdata->getMass();
            tag = h_tag.data[idx];
            }
        else
            {
            pidx = h_embed_idx->data[idx-N_mpcd];
            const Scalar4 vel_mass = h_vel_embed->data[pidx];
            mass = vel_mass.w;
            tag = h_tag_embed->data[pidx];
            }

        // draw random velocities from normal distribution
        hoomd::detail::Saru rng(tag, timestep, m_seed);
        mpcd::detail::NormalGenerator<Scalar,true> gen;
        const Scalar3 vel = fast::sqrt(T/mass) * make_scalar3(gen(rng), gen(rng), gen(rng));

        // save out velocities
        if (idx < N_mpcd)
            {
            h_vel.data[pidx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
            }
        else
            {
            h_vel_embed->data[pidx] = make_scalar4(vel.x, vel.y, vel.z, mass);
            }
        }
    }

void mpcd::ATCollisionMethod::applyVelocities()
    {
    // mpcd particle data
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel_alt(m_mpcd_pdata->getAltVelocities(), access_location::host, access_mode::read);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // embedded particle data
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_idx;
    std::unique_ptr< ArrayHandle<Scalar4> > h_vel_embed;
    std::unique_ptr< ArrayHandle<Scalar4> > h_vel_alt_embed;
    std::unique_ptr< ArrayHandle<unsigned int> > h_embed_cell_ids;
    if (m_embed_group)
        {
        h_embed_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(), access_location::host, access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(), access_location::host, access_mode::readwrite));
        h_vel_alt_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getAltVelocities(), access_location::host, access_mode::read));
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroupCellIds(), access_location::host, access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    ArrayHandle<double4> h_cell_vel(m_thermo->getCellVelocities(), access_location::host, access_mode::read);
    ArrayHandle<double4> h_rand_vel(m_rand_thermo->getCellVelocities(), access_location::host, access_mode::read);

    for (unsigned int idx=0; idx < N_tot; ++idx)
        {
        unsigned int cell, pidx;
        Scalar4 vel_rand;
        if (idx < N_mpcd)
            {
            pidx = idx;
            const Scalar4 vel_cell = h_vel.data[idx];
            cell = __scalar_as_int(vel_cell.w);
            vel_rand = h_vel_alt.data[idx];
            }
        else
            {
            pidx = h_embed_idx->data[idx-N_mpcd];
            cell = h_embed_cell_ids->data[idx-N_mpcd];
            vel_rand = h_vel_alt_embed->data[pidx];
            }

        // load cell data
        const double4 v_c = h_cell_vel.data[cell];
        const double4 vrand_c = h_rand_vel.data[cell];

        // compute new velocity using the cell + the random draw
        const Scalar3 vnew = make_scalar3(v_c.x - vrand_c.x + vel_rand.x,
                                          v_c.y - vrand_c.y + vel_rand.y,
                                          v_c.z - vrand_c.z + vel_rand.z);

        if (idx < N_mpcd)
            {
            h_vel.data[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, __int_as_scalar(cell));
            }
        else
            {
            h_vel_embed->data[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, vel_rand.w);
            }
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ATCollisionMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::ATCollisionMethod, std::shared_ptr<mpcd::ATCollisionMethod> >
        (m, "ATCollisionMethod", py::base<mpcd::CollisionMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      unsigned int,
                      unsigned int,
                      int,
                      unsigned int,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<::Variant>>())
        .def("setTemperature", &mpcd::ATCollisionMethod::setTemperature)
    ;
    }
