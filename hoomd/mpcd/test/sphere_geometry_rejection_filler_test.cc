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
void sphere_rejection_fill_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(20.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        std::shared_ptr<mpcd::ParticleDataSnapshot> mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(1);

        mpcd_snap->position[0] = vec3<Scalar>(1,-2,3);
        mpcd_snap->velocity[0] = vec3<Scalar>(123, 456, 789);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    auto pdata = mpcd_sys->getParticleData();
    mpcd_sys->getCellList()->setCellSize(1.0);
    // we should have no virtual particle in the system at this point.
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);

    // create a spherical confinement of radius 5.0
    Scalar r=5.0;
    auto sphere = std::make_shared<const mpcd::detail::SphereGeometry>(r, mpcd::detail::boundary::no_slip);
    std::shared_ptr<::Variant> kT = std::make_shared<::VariantConst>(1.5);
    std::shared_ptr<mpcd::RejectionVirtualParticleFiller<mpcd::detail::SphereGeometry>> filler = std::make_shared<F>(mpcd_sys, 2.0, 1, kT, 42, sphere);

    /*
     * Test basic filling up for this cell list
     */
    unsigned int Nfill_0(0);
    std::cout << "RUNNING THE FILLER FOR THE FIRST TIME" << "\n";
    filler->fill(0);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        // ensure first particle did not get overwritten
        CHECK_CLOSE(h_pos.data[0].x, 1, tol_small);
        CHECK_CLOSE(h_pos.data[0].y, -2, tol_small);
        CHECK_CLOSE(h_pos.data[0].z,  3, tol_small);
        CHECK_CLOSE(h_vel.data[0].x, 123, tol_small);
        CHECK_CLOSE(h_vel.data[0].y, 456, tol_small);
        CHECK_CLOSE(h_vel.data[0].z, 789, tol_small);
        UP_ASSERT_EQUAL(h_tag.data[0], 0);

        // check if the particles have been placed outside the confinement
        unsigned int N_out(0);
        for (unsigned int i=pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);
            // type should be set
            UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[i].w), 1);

            Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r*r)
                ++N_out;
            }
        std::cout << "Number of virtual particles outside confinement after first time filling = "<< N_out << "\n";
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        Nfill_0 = N_out;
        }

    /*
     * Fill the volume again, which should approximately double the number of virtual particles
     */
    std::cout << "RUNNING THE FILLER FOR THE SECOND TIME" << "\n";
    filler->fill(1);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        // check if the particles have been placed outside the confinement
        unsigned int N_out(0);
        for (unsigned int i=pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);
            // type should be set
            UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[i].w), 1);

            Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r*r)
                ++N_out;
            }
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        UP_ASSERT_GREATER(N_out, Nfill_0);
        }

    /*
     * Test the average properties of the virtual particles.
     */
    // initialize variables for storing avg data
    Scalar N_avg(0);
    Scalar3 vel_avg_net = make_scalar3(0,0,0);
    Scalar T_avg(0);
    // repeat filling 100 times
    unsigned int itr(100);
    for (unsigned int t=0; t<itr; ++t)
        {
        pdata->removeVirtualParticles();
        filler->fill(2+t);

        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);

        // local variables
        unsigned int N_out(0);
        Scalar temp(0);
        Scalar3 vel_avg = make_scalar3(0,0,0);

        for (unsigned int i=pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar3 vel = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r*r)
                ++N_out;
            temp += dot(vel, vel);
            vel_avg += vel;
            }

        temp /= (3*(N_out-1));
        vel_avg_net += vel_avg/N_out;
        // Check whether all virtual particles are outside the sphere
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        N_avg += N_out;
        T_avg += temp;

        }
    N_avg /= itr;
    T_avg /= itr;
    vel_avg_net /= itr;

    /*
    * Expected number of virtual particles = int( density * volume outside sphere )
    * volume outside sphere = sim-box volume - sphere volume
    * N_exptd = int(density*(L^3 - 4*pi*r^3/3))
    *         = 14952
    */
    UP_ASSERT_CLOSE(N_avg, 14952.0, 2);

    CHECK_SMALL(vel_avg_net.x, tol_small);
    CHECK_SMALL(vel_avg_net.y, tol_small);
    CHECK_SMALL(vel_avg_net.z, tol_small);
    CHECK_CLOSE(T_avg, 1.5, tol_small);
    }

UP_TEST( sphere_rejection_fill_basic )
    {
    sphere_rejection_fill_basic_test<mpcd::RejectionVirtualParticleFiller<mpcd::detail::SphereGeometry>>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_CUDA
UP_TEST( sphere_rejection_fill_basic_gpu )
    {
    sphere_rejection_fill_basic_test<mpcd::RejectionVirtualParticleFillerGPU<mpcd::detail::SphereGeometry>>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_CUDA