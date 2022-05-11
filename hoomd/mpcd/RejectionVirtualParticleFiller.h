
#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to MPCD particle data for "sphere" geometry
/*!
 * <detailed description needed>
*/

template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
        //! Constructor
        /*!
         * \param sysdata - MPCD system data
         * \param density - number density of MPCD solvent
         * \param type - particle fill type
         * \param T - Temperature
         * \param seed - seed for PRNG
         * \param geom - confinement geometry
         */
        RejectionVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                       Scalar density,
                                       unsigned int type,
                                       std::shared_ptr<::Variant> T,
                                       unsigned int seed,
                                       std::shared_ptr<const Geometry> geom)
        : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
        {
        m_exec_conf->msg->notice(5) << "constructing MPCD RejectionVirtualParticleFiller" << std::endl;
        }

        //! Destructor
        ~RejectionVirtualParticleFiller()
        {
        m_exec_conf->msg->notice(5) << "Destroying MPCD RejectionVirtualParticleFiller" << std::endl;
        }

        //! Get the streaming geometry
        std::shared_ptr<const Geometry> getGeometry() const
            {
            return m_geom;
            }

        //! Set the streaming geometry
        void setGeometry(std::shared_ptr<const Geometry> geom)
            {
            m_geom = geom;
            }

        //! Fill the particles outside the confinement
        void fill(unsigned int timestep);

    protected:
        std::shared_ptr<const Geometry> m_geom;
        GPUArray<Scalar4> m_tmp_pos;
        GPUArray<Scalar4> m_tmp_velTag;
    };


template<class Geometry>
void RejectionVirtualParticleFiller<Geometry>::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int NVirtMax = round(m_density*box.getVolume());

    // Step 1: Create temporary GPUArrays to draw Particles locally using the worst case estimate for number
    //         number of particles.
    if (NVirtMax > m_tmp_pos.getNumElements())
        {
        GPUArray<Scalar4> tmp_pos(NVirtMax, m_exec_conf);
        GPUArray<Scalar4> tmp_velTag(NVirtMax, m_exec_conf);
        m_tmp_pos.swap(tmp_pos);
        m_tmp_velTag.swap(tmp_velTag);
        }
    ArrayHandle<Scalar4> h_tmp_pos(m_tmp_pos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_tmp_velTag(m_tmp_velTag, access_location::host, access_mode::overwrite);

    // Step 2: Draw the particles and assign velocities simultaneously by using temporary memory
    unsigned int pidx = 0;
    unsigned int tag = computeFirstTag(&pidx);

    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    for (unsigned int i=0; i < NVirtMax; ++i)
        {
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::RejectionFiller, m_seed, timestep, tag);

        Scalar3 particle = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));

        if (m_geom->isOutside(particle))
            {
            h_tmp_pos.data[pidx] = make_scalar4(particle.x,
                                                particle.y,
                                                particle.z,
                                                __int_as_scalar(m_type));

            hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
            Scalar3 vel;
            gen(vel.x, vel.y, rng);
            vel.z = gen(rng);
            h_tmp_velTag.data[pidx++] = make_scalar4(vel.x,
                                                     vel.y,
                                                     vel.z,
                                                     __int_as_scalar(tag));

            tag++;
            }
        }

    // Allocate memory for the new virtual particles.
    const unsigned int first_idx = m_mpcd_pdata->addVirtualParticles(pidx);

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    // copy the temporary data to permanent data
    for (unsigned int i=0; i < pidx; ++i)
        {
        const unsigned int realidx = first_idx + i;
        // positions
        h_pos.data[realidx] = h_tmp_pos.data[i];
        // velocity
        Scalar4 swp = h_tmp_velTag.data[i];
        h_vel.data[realidx] = make_scalar4(swp.x,
                                           swp.y,
                                           swp.z,
                                           __int_as_scalar(mpcd::detail::NO_CELL));
        // tags
        h_tag.data[realidx] = (unsigned int)swp.w;
        }
    }

namespace detail
{
//! Export RejectionVirtualParticleFiller to python
template<class Geometry>
void export_RejectionVirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = Geometry::getName() + "RejectionFiller";
    py::class_<mpcd::RejectionVirtualParticleFiller<Geometry>, std::shared_ptr<mpcd::RejectionVirtualParticleFiller<Geometry>>>
        (m, name.c_str(), py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
             Scalar,
             unsigned int,
             std::shared_ptr<::Variant>,
             unsigned int,
             std::shared_ptr<const Geometry>>())
        .def("setGeometry", &mpcd::RejectionVirtualParticleFiller<Geometry>::setGeometry)
        .def("getGeometry", &mpcd::RejectionVirtualParticleFiller<Geometry>::getGeometry);
    }
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_FILLER_H_
