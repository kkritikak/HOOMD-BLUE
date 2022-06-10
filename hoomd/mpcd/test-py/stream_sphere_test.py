
import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd sphere streaming geometry
class mpcd_stream_sphere_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=4)

        # particle 1: Hits the wall in the second streaming step, gets reflected accordingly
        # particle 2: Always inside the sphere, so no reflection by the BC
        # particle 3: Hits the wall normally and gets reflected back.
        # particle 4: Lands almost exactly on the sphere surface and needs to be backtracked one complete step

        snap.particles.position[:] = [[2.85,0.895,np.sqrt(6)+0.075],
                                      [0.,0.,0.],
                                      list(0.965*np.array([-1.,-2.,np.sqrt(11)])),
                                      [1.92,-1.96,-np.sqrt(8)+0.05]]
        snap.particles.velocity[:] = [[1.,0.7,-0.5],
                                      [-1.,-1.,-1.],
                                      list(0.25*np.array([-1.,-2.,np.sqrt(11)])),
                                      [0.8,-0.4,-0.5]]
        self.s = mpcd.init.read_snapshot(snap)

        mpcd.integrator(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.stream.sphere(R=4., boundary="no_slip", period=2)

    # test for setting parameters
    def test_set_params(self):
        sphere = mpcd.stream.sphere(R=4.)
        self.assertAlmostEqual(sphere.R, 4.)
        self.assertEqual(sphere.boundary, "no_slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 4.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change R and also ensure other parameters stay the same
        sphere.set_params(R=2.)
        self.assertAlmostEqual(sphere.R, 2.)
        self.assertEqual(sphere.boundary, "no_slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 2.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change BCs
        sphere.set_params(boundary="slip")
        self.assertAlmostEqual(sphere.R, 2.)
        self.assertEqual(sphere.boundary, "slip")
        self.assertAlmostEqual(sphere._cpp.geometry.getR(), 2.)
        self.assertEqual(sphere._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        sphere = mpcd.stream.sphere(R=4.)
        sphere.set_params(boundary="no_slip")
        sphere.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            sphere.set_params(boundary="invalid")

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
        mpcd.stream.sphere(R=4.)

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [2.95,0.965,np.sqrt(6)+0.025])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,0.7,-0.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-0.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.99*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], 0.25*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.position[3], [2.,-2.,-np.sqrt(8)])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [0.8,-0.4,-0.5])

        # take another step where one particle will now reflect from the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [2.95,0.965,np.sqrt(6)+0.025])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,-0.7,0.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-0.2])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.985*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], -0.25*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.position[3], [1.92,-1.96,-np.sqrt(8)+0.05])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [-0.8,0.4,0.5])

        # take another step where both particles are streaming only
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [2.85,0.895,np.sqrt(6)+0.075])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,-0.7,0.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.3,-0.3,-0.3])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.96*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], -0.25*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.position[3], [1.84,-1.92,-np.sqrt(8)+0.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [-0.8,0.4,0.5])

    # test basic stepping behaviour with slip boundary conditions
    def test_step_slip(self):
        mpcd.stream.sphere(R=4., boundary="slip")

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [2.95,0.965,np.sqrt(6)+0.025])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,0.7,-0.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-0.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1., -1., -1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.99*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], 0.25*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.position[3], [2.,-2.,-np.sqrt(8)])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [0.8,-0.4,-0.5])

        # take another step where one particle will now hit the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            # point of contact
            r1_before = np.array([3.,1.,np.sqrt(6)])
            # velocity before reflection
            v1_before = np.array([1.,0.7,-0.5])
            # velocity after reflection
            v1_after = v1_before - 1/8.*np.dot(v1_before,r1_before)*r1_before
            # position after reflection
            r1_after = r1_before+v1_after*0.05
            np.testing.assert_array_almost_equal(snap.particles.position[0], r1_after)
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], v1_after)
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-0.2])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1., -1., -1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.985*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], -0.25*np.array([-1.,-2.,np.sqrt(11)]))
            r2_before = np.array([2.,-2.,-np.sqrt(8)])
            v2_before = np.array([0.8,-0.4,-0.5])
            v2_after = v2_before - 1./8.*np.dot(v2_before,r2_before)*r2_before
            r2_after = r2_before+v2_after*0.1
            np.testing.assert_array_almost_equal(snap.particles.position[3], r2_after)
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], v2_after)

        # take another step where both particles are streaming only
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            # one step streaming
            r1_after += v1_after*0.1
            np.testing.assert_array_almost_equal(snap.particles.position[0], r1_after)
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], v1_after)
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.3,-0.3,-0.3])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], 0.96*np.array([-1.,-2.,np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], -0.25*np.array([-1.,-2.,np.sqrt(11)]))
            r2_after += v2_after*0.1
            np.testing.assert_array_almost_equal(snap.particles.position[3], r2_after)
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], v2_after)

    # test that setting the sphere size too large raises an error
    def test_validate_box(self):
        # initial configuration is invalid
        sphere = mpcd.stream.sphere(R=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        # now it should be valid
        sphere.set_params(R=4.)
        hoomd.run(2)

        # make sure we can invalidate it again
        sphere.set_params(R=4.1)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
        sphere = mpcd.stream.sphere(R=3.8)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        sphere.set_params(R=3.95)
        hoomd.run(1)

    # test that virtual particle filler can be attached, removed and updated
    def test_filler(self):
        # initialization of a filler
        sphere = mpcd.stream.sphere(R=4.0)
        sphere.set_filler(density=5.0, kT=1.0, seed=79, type='A')
        self.assertTrue(sphere._filler is not None)

        # run should be able to setup the filler, although this all happens silently
        hoomd.run(2)

        # changing the geometry should still be OK with a run
        sphere.set_params(R=3.99)
        hoomd.run(1)

        # changing filler should be allowed
        sphere.set_filler(density=10.0, kT=1.5, seed=51)
        self.assertTrue(sphere._filler is not None)
        hoomd.run(1)

        # assert an error is raised if we set a bad particle type
        with self.assertRaises(RuntimeError):
            sphere.set_filler(density=5., kT=1.0, seed=42, type='B')

        # assert an error is raised if we set a bad density
        with self.assertRaises(RuntimeError):
            sphere.set_filler(density=-1.0, kT=1.0, seed=42)

        # removing the filler should still allow a run
        sphere.remove_filler()
        self.assertTrue(sphere._filler is None)
        hoomd.run(1)

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
