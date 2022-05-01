
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
        public void fill(unsigned int timestep);

    protected:
        std::shared_ptr<const Geometry> m_geom;

    };


template<class Geometry>
void RejectionVirtualParticleFiller<Geometry>::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int m_NVirtMax = round(m_density*box.getVolume());

    // Step 1: Create temporary GPUArrays to draw Particles locally using the worst case estimate for number
    //         number of particles.
    GPUArray<Scalar4> pos_loc(m_NVirtMax, m_exec_conf);

    // Step 2: Draw the particles.
    unsigned int pidx = 0;
    for (unsigned int i=0; i < m_NVirtMax; ++i)
        {
        // TODO: Currently just using the constant for SlitGeometryFiller which needs to be changed
        // NOTE: particle tags neglected for the RNG here since we don't know them.
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::SlitGeometryFiller, m_seed, timestep);

        Scalar3 particle = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));

        if (m_geom->isOutside(particle))
            {
            pos_loc[pidx] = make_scalar4(vel.x,
                                         vel.y,
                                         vel.z,
                                         __int_as_scalar(mpcd::detail::NO_CELL));
            pidx += 1;
            }
        }

    // in mpi, do a prefix scan on the tag offset in this range
    // then shift the first tag by the current number of particles, which ensures a compact tag array
    m_first_tag = 0;
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        // scan the number to fill to get the tag range I own
        MPI_Exscan(&m_N_fill, &m_first_tag, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI
    m_first_tag += m_mpcd_pdata->getNGlobal() + m_mpcd_pdata->getNVirtualGlobal();

    // Allocate memory for the new virtual particles.
    m_mpcd_pdata->addVirtualParticles(pidx);

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    // index to start filling from
    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - pidx;
    for (unsigned int i=0; i < pidx; ++i);
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::SlitGeometryFiller, m_seed, tag, timestep);

        const unsigned int realidx = first_idx + i;
        h_pos.data[realidx] = pos_loc[i];

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        h_vel.data[pidx] = make_scalar4(vel.x,
                                        vel.y,
                                        vel.z,
                                        __int_as_scalar(mpcd::detail::NO_CELL));

        h_tag.data[realidx] = tag;
        }
    }

namespace detail
{
//! Export RejectionVirtualParticleFiller to python
template<class Geometry>
void export_RejectionVirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::RejectionVirtualParticleFiller<Geometry>, std::shared_ptr<mpcd::RejectionVirtualParticleFiller<Geometry>>>
        (m, "RejectionFiller", py::base<mpcd::VirtualParticleFiller>())
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
