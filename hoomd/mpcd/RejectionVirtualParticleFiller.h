
#ifndef MPCD_REJECTION_FILLER_H_
#define MPCD_REJECTION_FILLER_H_

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

    protected:
        std::shared_ptr<const Geometry> m_geom;

        //! Fill the particles outside the confinement
        void fill(unsigned int timestep);
    };


template<class Geometry>
void RejectionVirtualParticleFiller<Geometry>::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int m_Lx = hi.x - lo.x;
    const unsigned int m_Ly = hi.y - lo.y;
    const unsigned int m_Lz = hi.z - lo.z;
    const unsigned int m_NVirtMax = m_density*(m_Lx*m_Ly*m_Lz);

    // Allocate memory
    // first remove any previously added memory for virtual particles
    // This is probably being done already, so that needs to be checked.
    m_mpcd_pdata->removeVirtualParticles();
    // Add N virtual particles as a worst case estimate
    m_mpcd_pdata->addVirtualParticles(m_NVirtMax);

    // Draw particles
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    // index to start filling from
    unsigned int pidx = m_mpcd_pdata->getN();
    for (unsigned int i=0; i < m_NVirtMax; ++i)
        {
        // TODO: Currently just using the constant for SlitGeometryFiller which needs to be changed.
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::SlitGeometryFiller, m_seed, timestep);

        Scalar3 tmp_pos = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                       hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));

        if (m_geom->isOutside(tmp_pos))
            {
            h_pos.data[pidx] = make_scalar4(tmp_pos.x,
                                            tmp_pos.y,
                                            tmp_pos.z,
                                            __int_as_scalar(m_type));

            hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
            Scalar3 vel;
            gen(vel.x, vel.y, rng);
            vel.z = gen(rng);
            h_vel.data[pidx] = make_scalar4(vel.x,
                                            vel.y,
                                            vel.z,
                                            __int_as_scalar(mpcd::detail::NO_CELL));

            pidx += 1;
            }
        }
    m_mpcd_pdata->setVirtualParticleNumber(pidx - m_mpcd_pdata->getN());
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
