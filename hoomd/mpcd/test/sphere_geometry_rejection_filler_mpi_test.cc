// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "hoomd/mpcd/RejectionVirtualParticleFiller.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.h"
#endif // ENABLE_CUDA
#include "hoomd/mpcd/SphereGeometry.h"

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

template<class F>
void sphere_rejection_fill_mpi_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(20.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf,snap->global_box.getL(),2,2,2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));

    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        std::shared_ptr<mpcd::ParticleDataSnapshot> mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(1);

        mpcd_snap->position[0] = vec3<Scalar>(1,-2,3);
        mpcd_snap->velocity[0] = vec3<Scalar>(123, 456, 789);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    auto pdata = mpcd_sys->getParticleData();
    // There should be no virtual particles in the system at this point
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);

    // create a spherical confinement of radius 5.0
    Scalar r=5.0;
    auto sphere = std::make_shared<const mpcd::detail::SphereGeometry>(r ,0.0 , mpcd::detail::boundary::no_slip);
    std::shared_ptr<::Variant> kT = std::make_shared<::VariantConst>(1.5);
    std::shared_ptr<mpcd::RejectionVirtualParticleFiller<mpcd::detail::SphereGeometry>> filler = std::make_shared<F>(mpcd_sys, 2.0, 1, kT, 63, sphere);

    /*
     * Test basic filling up for this cell list
     */
    filler->fill(0);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        const BoxDim& box = sysdef->getParticleData()->getBox();
        // check if the virtual particles are outside the sphere
        unsigned int N_out(0);
        for (unsigned int i = 0; i < pdata->getNVirtual(); ++i)
            {
            const unsigned int idx = pdata->getN() + i;
            // check if the virtual particles are in the box
            UP_ASSERT(h_pos.data[idx].x >= box.getLo().x && h_pos.data[idx].x < box.getHi().x);
            UP_ASSERT(h_pos.data[idx].y >= box.getLo().y && h_pos.data[idx].y < box.getHi().y);
            UP_ASSERT(h_pos.data[idx].z >= box.getLo().z && h_pos.data[idx].z < box.getHi().z);

            Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r*r)
                ++N_out;
            }
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        }

    /*
    * Test avg. number of virtual particles on each rank by filling up the system N_samples(=500) times
    */
    Scalar N_avg_rank(0);
    Scalar N_avg_global(0);
    unsigned int itr(500);
    for (unsigned int t=0; t<itr; ++t)
        {
        pdata->removeVirtualParticles();
        UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);
        filler->fill(1+t);

        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);

        N_avg_rank += pdata->getNVirtual();
        N_avg_global += pdata->getNVirtualGlobal();
        }
    N_avg_rank /= itr;
    N_avg_global /= itr;

    /*
     * Expected number of virtual particles = int( density * volume outside sphere )
     * volume outside sphere = sim-box volume - sphere volume
     * N_exptd = int(density*(L^3 - 4*pi*r^3/3))
     *         = 14952
     * 8 CPUs -> each CPU has equal volume to fill with virtual particles
     * Therefore, N_exptd_rank = 14952/8 = 1869
    */
    UP_ASSERT_CLOSE(N_avg_rank, Scalar(1869), tol_small);
    UP_ASSERT_CLOSE(N_avg_global, Scalar(14952), tol_small);
    }

UP_TEST( sphere_rejection_fill_mpi )
    {
    sphere_rejection_fill_mpi_test<mpcd::RejectionVirtualParticleFiller<mpcd::detail::SphereGeometry>>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_CUDA
UP_TEST( sphere_rejection_fill_mpi_gpu )
    {
    sphere_rejection_fill_mpi_test<mpcd::RejectionVirtualParticleFillerGPU<mpcd::detail::SphereGeometry>>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_CUDA
