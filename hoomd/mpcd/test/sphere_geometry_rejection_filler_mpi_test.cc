
#include "hoomd/mpcd/RejectionVirtualParticleFiller.h"

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
    auto sphere = std::make_shared<const mpcd::detail::SphereGeometry>(r, mpcd::detail::boundary::no_slip);
    std::shared_ptr<::Variant> kT = std::make_shared<::VariantConst>(1.5);
    std::shared_ptr<mpcd::RejectionVirtualParticleFiller<mpcd::detail::SphereGeometry>> filler = std::make_shared<F>(mpcd_sys, 2.0, 1, kT, 42, sphere);

    /*
     * Test basic filling up for this cell list
     */
    unsigned int Nfill_0(0);
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
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        Nfill_0 = N_out;
        }

    }