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
#include "RejectionVirtualParticleFillerGPU.cuh"

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
        : mpcd::RejectionVirtualParticleFiller<Geometry>(sysdata, density, type, T, seed, geom),
        m_track_bounded_particles(this->m_exec_conf), m_compact_pos(this->m_exec_conf), m_compact_vel(this->m_exec_conf)
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
        GPUArray<Scalar4> m_compact_pos;
        GPUArray<Scalar4> m_compact_vel;

    private:
        std::unique_ptr<::Autotuner> m_tuner;   //!< Autotuner for drawing particles\
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
        GPUArray<Scalar4> compact_pos(N_virt_max, this->m_exec_conf);
        GPUArray<Scalar4> compact_vel(N_virt_max, this->m_exec_conf);
        m_tmp_pos.swap(tmp_pos);
        m_tmp_vel.swap(tmp_vel);
        m_track_bounded_particles.swap(track_bounded_particles);
        m_compact_pos.swap(compact_pos);
        m_compact_vel.swap(compact_vel);
        }
    ArrayHandle<Scalar4> d_tmp_pos(m_tmp_pos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tmp_vel(m_tmp_vel, access_location::host, access_mode::overwrite);
    ArrayHandle<bool> d_track_bounded_particles(m_track_bounded_particles, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> d_compact_pos(m_compact_pos. access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> d_compact_vel(m_compact_vel. access_location::host, access_mode::overwrite);

    // Step 2: Draw particle positions and velocities in parallel on GPU
    unsigned int first_tag = computeFirstTag(N_virt_max);

    mpcd::gpu::draw_virtual_particles_args_t args(d_tmp_pos.data,
                                                  d_tmp_vel.data,
                                                  d_track_bounded_particles,
                                                  lo, hi,
                                                  first_tag,
                                                  m_filler_id,
                                                  m_type,
                                                  N_virt_max,
                                                  timestep,
                                                  m_seed,
                                                  m_tuner->getParam());

    m_tuner->begin();
    mpcd::gpu::draw_virtual_particles<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

    unsigned int n_pos_selected(0);
    mpcd::gpu::compact_data_arrays(d_tmp_pos.data,
                                   d_track_bounded_particles,
                                   N_virt_max,
                                   d_compact_pos,
                                   n_pos_selected)

    m_tuner->end();


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