
#include "RejectionVirtualParticleFiller.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

mpcd::RejectionVirtualParticleFiller::RejectionVirtualParticleFiller(
                                       std::shared_ptr<mpcd::SystemData> sysdata,
                                       Scalar density,
                                       unsigned int type,
                                       std::shared_ptr<::Variant> T,
                                       unsigned int seed,
                                       std::shared_ptr<const Geometry> geom);
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "constructing MPCD RejectionVirtualParticleFiller" << std::endl;
    }

mpcd::RejectionVirtualParticleFiller::~RejectionVirtualParticleFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD RejectionVirtualParticleFiller" << std::endl;
    }

void mpcd::RejectionVirtualParticleFiller::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    const unsigned int m_L = hi - lo;
    const unsigned int m_NVirtMax = m_density*(m_L*m_L*m_L);

    // Allocate memory
    // first remove any previously added memory for virtual particles
    // This is probably being done already, so that needs to be checked.
    m_mpcd_pdata->removeVirtualParticles();
    // Add N virtual particles as a worst case estimate
    m_mpcd_pdata->addVirtalParticles(m_NVirtMax);

    // Draw particles
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    // index to start filling from
    const unsigned int pidx = m_mpcd_pdata->getN();
    for (unsigned int i=0; i<m_NVirtMax; ++i)
        {
        /*TODO: Currently just using the constant for SlitGeometryFiller which needs to be changed in the final version
          */
        hoomd::RandomGenerator rng(hoomd::RNGIdentifiers::SlitGeometryFiller, m_seed, timestep);

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
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_RejectionVirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::RejectionVirtualParticleFiller, std::shared_ptr<mpcd::RejectionVirtualParticleFiller>>
        (m, "RejectionFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
             Scalar,
             unsigned int,
             std::shared_ptr<::Variant>,
             unsigned int,
             std::shared_ptr<const Geometry>>())
        .def("setGeometry", &mpcd::RejectionVirtualParticleFiller::setGeometry);
    }
