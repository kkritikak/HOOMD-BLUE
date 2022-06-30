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

#include "RejectionVirtualParticleFiller.h"
#include "RejectionVirtualParticleFillerGPU.cuh"

#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to the MPCD particle data for SphereGeometry using the GPU
template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFillerGPU : public mpcd::RejectionVirtualParticleFiller<Geometry>
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
        m_track_bounded_particles(this->m_exec_conf), m_compact_idxs(this->m_exec_conf),
        m_temp_storage(this->m_exec_conf)
        {
        m_tuner1.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_rejection_filler_draw_particles" + Geometry::getName(), this->m_exec_conf));
        m_tuner2.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_rejection_filler_tag_particles" + Geometry::getName(), this->m_exec_conf));
        }

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::RejectionVirtualParticleFiller<Geometry>::setAutotunerParams(enable, period);

            m_tuner1->setEnabled(enable); m_tuner1->setPeriod(period);
            m_tuner2->setEnabled(enable); m_tuner2->setPeriod(period);
            }

    protected:
        //! Fill the volume outside the confinement
        void fill(unsigned int timestep);
        GPUArray<bool> m_track_bounded_particles;
        GPUArray<unsigned int> m_compact_idxs;
        GPUArray<unsigned int> m_temp_storage;

    private:
        std::unique_ptr<::Autotuner> m_tuner1;   //!< Autotuner for drawing particles
        std::unique_ptr<::Autotuner> m_tuner2;   //!< Autotuner for particle tagging
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
        GPUArray<unsigned int> compact_idxs(N_virt_max, this->m_exec_conf);
        GPUArray<unsigned int> temp_storage(N_virt_max, this->m_exec_conf);
        this->m_tmp_pos.swap(tmp_pos);
        this->m_tmp_vel.swap(tmp_vel);
        m_track_bounded_particles.swap(track_bounded_particles);
        m_compact_idxs.swap(compact_idxs);
        m_temp_storage.swap(temp_storage);
        }
    ArrayHandle<Scalar4> d_tmp_pos(this->m_tmp_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tmp_vel(this->m_tmp_vel, access_location::device, access_mode::overwrite);
    ArrayHandle<bool> d_track_bounded_particles(m_track_bounded_particles, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_compact_idxs(m_compact_idxs, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_temp_storage(m_temp_storage, access_location::device, access_mode::overwrite);

    // Step 2: Draw particle positions and velocities in parallel on GPU
    unsigned int first_tag = this->computeFirstTag(N_virt_max);

    const Scalar vel_factor = fast::sqrt(this->m_T->getValue(timestep) / this->m_mpcd_pdata->getMass());

    mpcd::gpu::draw_virtual_particles_args_t args(d_tmp_pos.data,
                                                  d_tmp_vel.data,
                                                  d_track_bounded_particles.data,
                                                  lo,
                                                  hi,
                                                  first_tag,
                                                  vel_factor,
                                                  this->m_filler_id,
                                                  this->m_type,
                                                  N_virt_max,
                                                  timestep,
                                                  this->m_seed,
                                                  m_tuner1->getParam());

    m_tuner1->begin();
    mpcd::gpu::draw_virtual_particles<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner1->end();

    unsigned int n_selected(0);
    mpcd::gpu::compact_indices(d_track_bounded_particles.data,
                               N_virt_max,
                               d_compact_idxs.data,
                               &n_selected,
                               d_temp_storage.data);

    // Compute the correct tags
    first_tag = this->computeFirstTag(N_virt_max);

    // Allocate memory for the new virtual particles.
    const unsigned int first_idx = this->m_mpcd_pdata->addVirtualParticles(n_selected);

    ArrayHandle<Scalar4> d_pos(this->m_mpcd_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(this->m_mpcd_pdata->getTags(), access_location::device, access_mode::readwrite);

    // Copy data from temporary arrays to permanent arrays
    m_tuner2->begin();
    mpcd::gpu::parallel_copy(d_compact_idxs.data, d_pos.data, d_vel.data, d_tag.data, d_tmp_pos.data, d_tmp_vel.data,
                             first_idx, first_tag, n_selected, m_tuner2->getParam());
    m_tuner2->end();
    }

namespace detail
{
//! Export RejectionVirtualParticleFillerGPU to python
template<class Geometry>
void export_RejectionVirtualParticleFillerGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = Geometry::getName() + "RejectionFillerGPU";
    py::class_<mpcd::RejectionVirtualParticleFillerGPU<Geometry>, std::shared_ptr<mpcd::RejectionVirtualParticleFillerGPU<Geometry>>>
        (m, name.c_str(), py::base<mpcd::RejectionVirtualParticleFiller<Geometry>>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
             Scalar,
             unsigned int,
             std::shared_ptr<::Variant>,
             unsigned int,
             std::shared_ptr<const Geometry>>());
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_