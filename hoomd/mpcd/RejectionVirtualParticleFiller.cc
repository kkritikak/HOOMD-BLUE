
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
    // Question: How are we regulating the number of particles inside the spherical confinement?
    //           How is it being done in the other geometries?
    //
    const unsigned int m_N = m_mpcd_pdata->getN();

    // Allocate memory
    // first remove any previously added memory for virtual particles
    m_mpcd_pdata->removeVirtualParticles();
    // Add



    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    for (unsigned int i=0; i<m_N; ++i)
        {
        /*TODO: Currently just using the constant for SlitGeometryFiller which needs to be changed in the final version
          */
        hoomd::RandomGenerator rng(hoomd::RNGIdentifiers::SlitGeometryFiller, m_seed, timestep);

        Scalar3 tmp_pos = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));

        // The way of accessing the geometry is most probably wrong.
        if (m_geom->isOutside(tmp_pos))
            {


            }


        }

    }