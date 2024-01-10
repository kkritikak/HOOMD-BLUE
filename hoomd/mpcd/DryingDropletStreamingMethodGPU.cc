// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/DryingDropletStreamingMethodGPU.cc
 * \brief Definition of mpcd::DryingDropletStreamingMethodGPU
 */

#include "DryingDropletStreamingMethodGPU.h"
#include "DryingDropletStreamingMethodGPU.cuh"

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
mpcd::DryingDropletStreamingMethodGPU::DryingDropletStreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                                       unsigned int cur_timestep,
                                                                       unsigned int period,
                                                                       int phase,
                                                                       std::shared_ptr<::Variant> R,
                                                                       mpcd::detail::boundary bc,
                                                                       Scalar density,
                                                                       unsigned int seed)
    : mpcd::ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>(sysdata, cur_timestep, period, phase, std::make_shared<mpcd::detail::SphereGeometry>(R->getValue(cur_timestep), 0.0, bc)),
      m_R(R), m_bc(bc), m_density(density), m_seed(seed),m_picks(this->m_exec_conf), m_removed(this->m_exec_conf), m_picker(m_sysdef, seed)
    {
    m_apply_picks_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_apply_picks", this->m_exec_conf));
    const Scalar start_R = this->m_R->getValue(cur_timestep);
    const unsigned int next_timestep = (m_next_timestep != cur_timestep) ? m_next_timestep : cur_timestep + m_period;
    const Scalar end_R = this->m_R->getValue(next_timestep);
    const Scalar V = (end_R - start_R) / (m_mpcd_dt * (next_timestep - cur_timestep) / m_period);
    this->m_geom = std::make_shared<mpcd::detail::SphereGeometry>(start_R, V, bc);
    }

/*!
 * \param timestep Current time to stream
 */
void mpcd::DryingDropletStreamingMethodGPU::stream(unsigned int timestep)
    {
    /* 
     * This code duplicates DryingDropletMethodStreamingMethod::stream.
     * See there for comments.
     */
    if(!this->peekStream(timestep)) return;

    if (this->m_validate_geom)
        {
        this->validate();
        this->m_validate_geom = false;
        }

    const Scalar start_R = this->m_R->getValue(timestep);
    const Scalar end_R = this->m_R->getValue(timestep + m_period);
    const Scalar V = (end_R - start_R)/(m_mpcd_dt);
    if (V > 0)
        {
        throw std::runtime_error("Droplet radius must decrease.");
        }

    this->m_geom = std::make_shared<mpcd::detail::SphereGeometry>(end_R, V, m_bc );
    if (this->m_filler)
        {
        this->m_filler->setGeometry(this->m_geom);
        }

    ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>::stream(timestep);

    const unsigned int N_end = std::round((4.*M_PI/3.)*end_R*end_R*end_R*m_density);
    const unsigned int N_global = this->m_mpcd_pdata->getNGlobal();
    const unsigned int N_evap = (N_end < N_global) ? N_global - N_end : 0;

    unsigned int Npick = 0;
    m_picker(m_picks, Npick, m_bounced, N_evap, timestep, this->m_mpcd_pdata->getN());

    // apply picks using the GPU
    const unsigned int mask = 1 << 1;     //!< Mask for setting additional bit in m_bounced
        {
        ArrayHandle<unsigned int> d_picks(this->m_picks, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_bounced(this->m_bounced, access_location::device, access_mode::readwrite);
        m_apply_picks_tuner->begin();
        mpcd::gpu::apply_picks(d_bounced.data,
                               d_picks.data,
                               mask,
                               Npick,
                               m_apply_picks_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_apply_picks_tuner->end();
        }

    this->m_mpcd_pdata->removeParticlesGPU(this->m_removed,
                                           this->m_bounced,
                                           mask,
                                           timestep);

    Scalar V_end = (4.*M_PI/3.)*end_R*end_R*end_R;
    Scalar currentdensity = this->m_mpcd_pdata->getNGlobal()/V_end;
    if (std::fabs(currentdensity - m_density) > Scalar(0.1) * m_density)
        {
        this->m_exec_conf->msg->warning() << "Solvent density changed to: " << currentdensity << std::endl;
        }
    }

void mpcd::detail::export_DryingDropletStreamingMethodGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::DryingDropletStreamingMethodGPU, mpcd::ConfinedStreamingMethodGPU<mpcd::detail::SphereGeometry>, std::shared_ptr<mpcd::DryingDropletStreamingMethodGPU>>(m, "DryingDropletStreamingMethodGPU")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, std::shared_ptr<::Variant>, boundary, Scalar, unsigned int>())
        .def_property("filler", &mpcd::DryingDropletStreamingMethodGPU::getFiller, &mpcd::DryingDropletStreamingMethodGPU::setFiller);
    }
