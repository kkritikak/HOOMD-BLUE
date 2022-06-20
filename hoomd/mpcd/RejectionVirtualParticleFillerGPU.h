// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.h
 * \brief Definition of virtual particle filler for mpcd::detail::SphereGeometry on the GPU. (for now)
 */

#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "RejectionVirtualParticlefiller.h"
#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to the MPCD particle data for SphereGeometry using the GPU
template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFillerGPU : public mpcd::RejectionVirtualParticleFiller
    {
    public:
        //! Constructor
        RejectionVirtualParticleFillerGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                          Scalar density,
                                          unsigned int type,
                                          std::shared_ptr<::Variant> T,
                                          unsigned int seed,
                                          std::shared_ptr<const Geometry> geom)
        : mpcd::RejectionVirtualParticleFiller<Geometry>(sysdata, density, type, T, seed, geom), m_track_bounded_particles(this->m_exec_conf)
        {
        m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_rejection_filler" + Geometry::getName(), this->m_exec_conf));
        }

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::RejectionVirtualParticleFiller::setAutotunerParams(enable, period);

            m_tuner->setEnabled(enable); m_tuner->setPeriod(period);
            }

    protected:
        //! Fill the volume outside the confinement
        void fill(unsigned int timestep);
        GPUArray<bool> m_track_bounded_particles;

    private:
        std::unique_ptr<::Autotuner> m_tuner;   //!< Autotuner for drawing particles
    };


template<class Geometry>
void RejectionVirtualParticleFillerGPU<Geometry>::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = this->m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int N_virt_max = round(this->m_density*box.getVolume());

    // Step 1: Create temporary arrays
    if (N_virt_max > this->m_tmp_pos.getNumElements())
        {
        GPUArray<Scalar4> tmp_pos(N_virt_max, this->m_exec_conf);
        GPUArray<Scalar4> tmp_vel(N_virt_max, this->m_exec_conf);
        GPUArray<bool> track_bounded_particles(N_virt_max, this->m_exec_conf);
        this->m_tmp_pos.swap(tmp_pos);
        this->m_tmp_vel.swap(tmp_vel);
        this->m_track_bounded_particles.swap(track_bounded_particles);
        }
    ArrayHandle<Scalar4> d_tmp_pos(this->m_tmp_pos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tmp_vel(this->m_tmp_vel, access_location::host, access_mode::overwrite);
    ArrayHandle<bool> d_track_bounded_particles(this->m_track_bounded_particles, access_location::host, access_mode::overwrite);
    
    // Step 2: Draw particle positions and velocities in parallel on GPU
    unsigned int first_tag = computeFirstTag(N_virt_max);

    m_tuner->begin();



    }



namespace detail
{
//! Export RejectionVirtualParticleFillerGPU to python
template<class Geometry>
void export_RejectionVirtualParticleFiller(pybind11::module& m)
    {

    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_