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

#include "hoomd/CachedAllocator.h"
#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

#include <iterator>

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
        m_keep_particles(this->m_exec_conf), m_keep_indices(this->m_exec_conf), m_num_keep(this->m_exec_conf)
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

    private:
        GPUArray<bool> m_keep_particles; // Track if the particles are in/out of bounds for geometry
        GPUArray<unsigned int> m_keep_indices; // Indices for particles out of bound for geometry
        GPUFlags<unsigned int> m_num_keep;
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
        this->m_tmp_pos.swap(tmp_pos);
        GPUArray<Scalar4> tmp_vel(N_virt_max, this->m_exec_conf);
        this->m_tmp_vel.swap(tmp_vel);
        GPUArray<bool> keep_particles(N_virt_max, this->m_exec_conf);
        m_keep_particles.swap(keep_particles);
        GPUArray<unsigned int> keep_indices(N_virt_max, this->m_exec_conf);
        m_keep_indices.swap(keep_indices);
        }
    ArrayHandle<Scalar4> d_tmp_pos(this->m_tmp_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tmp_vel(this->m_tmp_vel, access_location::device, access_mode::overwrite);
    ArrayHandle<bool> d_keep_particles(m_keep_particles, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_keep_indices(m_keep_indices, access_location::device, access_mode::overwrite);

    // Step 2: Draw particle positions and velocities in parallel on GPU
    unsigned int first_tag = this->computeFirstTag(N_virt_max);

    const Scalar vel_factor = fast::sqrt(this->m_T->getValue(timestep) / this->m_mpcd_pdata->getMass());

    mpcd::gpu::draw_virtual_particles_args_t args(d_tmp_pos.data,
                                                  d_tmp_vel.data,
                                                  d_keep_particles.data,
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

    // Pick (N=box_vol*solvPartDensity) particles and velocities randomly in the box
    m_tuner1->begin();
    mpcd::gpu::draw_virtual_particles<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner1->end();


    // compact particle indices
    {
    // 1. Determine storage requirement by using a NULL ptr as input to cub function
    void* d_tmp_storage = NULL;
    size_t tmp_storage_bytes = 0;
    mpcd::gpu::compact_virtual_particle_indices(d_tmp_storage,
                                                tmp_storage_bytes,
                                                d_keep_particles.data,
                                                N_virt_max,
                                                d_keep_indices.data,
                                                m_num_keep.getDeviceFlags());
    // 2. Check temporary storage availability
    ScopedAllocation<unsigned char> d_tmp_alloc(this->m_exec_conf->getCachedAllocator(),
                                                (tmp_storage_bytes > 0) ? tmp_storage_bytes : 1);
    d_tmp_storage = (void*)d_tmp_alloc();
    // 3. Run selection
    mpcd::gpu::compact_virtual_particle_indices(d_tmp_storage,
                                                tmp_storage_bytes,
                                                d_keep_particles.data,
                                                N_virt_max,
                                                d_keep_indices.data,
                                                m_num_keep.getDeviceFlags());
    }
    unsigned int n_selected = m_num_keep.readFlags();

    // Compute the correct tags
    first_tag = this->computeFirstTag(n_selected);

    // Allocate memory for the new virtual particles.
    const unsigned int first_idx = this->m_mpcd_pdata->addVirtualParticles(n_selected);

    ArrayHandle<Scalar4> d_pos(this->m_mpcd_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(this->m_mpcd_pdata->getTags(), access_location::device, access_mode::readwrite);

    // Copy data from temporary arrays to permanent arrays
    m_tuner2->begin();
    mpcd::gpu::copy_virtual_particles(d_keep_indices.data, d_pos.data, d_vel.data, d_tag.data, d_tmp_pos.data, d_tmp_vel.data,
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
