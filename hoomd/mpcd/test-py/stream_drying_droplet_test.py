import unittest
import numpy as np
import hoomd
from hoomd import mpcd

# unit tests for mpcd sphere streaming geometry
class mpcd_stream_sphere_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()
        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=105.)))

        # initialize the system from the starting snapshot
        self.density = 5.
        R = 50.
        Pe = 5.
        dt = 0.001
        alpha = 8*np.pi*Pe
        self.num_particles = int(self.density*(4./3.)*np.pi*R*R*R)

        #creating a variant for R
        def variant_R(t):
            R_t = np.sqrt(R*R - (alpha*t*dt/(4.*np.pi)))
            return R_t
        my_variant = []
        for i in range(150):
            my_variant.append((i , variant_R(i)))

        self.R1 = hoomd.variant.linear_interp(points = my_variant)

        #making snapshot
        self.mpcd_sys = hoomd.mpcd.init.make_random(N=self.num_particles, kT=1.0, seed=42)
        snap = self.mpcd_sys.take_snapshot()

        # modify snapshot particles positions so they lie inside droplet
        if hoomd.comm.get_rank() == 0:
            particles = []
            while len(particles) < snap.particles.N:
                x = np.random.uniform(-R, R)
                y = np.random.uniform(-R, R)
                z = np.random.uniform(-R, R)
                if x**2 + y**2 + z**2 < R**2:
                    particles.append((x, y, z))
            snap.particles.position[:] = particles[:]

        #restoring the snapshot
        self.mpcd_sys.restore_snapshot(snap)

        mpcd.integrator(dt=dt)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.stream.drying_droplet(R=self.R1, density=self.density, period=1, boundary="no_slip", seed=1991)

    # test that if the droplet size too large raises an error
    def test_validate_box(self):
        # initial configuration is invalid
        Rt = hoomd.variant.linear_interp(points = [(0,55.),(10,55.)])
        mpcd.stream.drying_droplet(R=Rt, density=self.density, period=1, boundary="no_slip", seed=1661)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
        Rt = hoomd.variant.linear_interp(points = [(0,40.),(10,40.)])
        mpcd.stream.drying_droplet(R=Rt, density=self.density, period=1, boundary="no_slip", seed=1441)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test for invalid boundary conditions being provided
    def test_bad_boundary(self):
        with self.assertRaises(ValueError):
            mpcd.stream.drying_droplet(R=self.R1, density=self.density, period=1, boundary="invalid", seed=1221)

    def test_for_Nbounced(self):
        # initializing the droplet
        self.mpcd_sys.sorter.set_period(period = 10)
        drying_sphere = mpcd.stream.drying_droplet(R=self.R1, density = self.density, period=1, boundary="no_slip", seed=1991)

        # take 1 step and seeing if number of particles are decreasing or not
        hoomd.run(1)
        snap = self.mpcd_sys.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            num_par_after1 = snap.particles.N
            self.assertLessEqual(num_par_after1, self.num_particles)

        # take another step and see if number of particles are decreasing again
        hoomd.run(1)
        snap = self.mpcd_sys.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            num_par_after2 = snap.particles.N
            self.assertLessEqual(num_par_after2, num_par_after1)

    # test that virtual particle filler can be attached, removed and updated
    def test_filler(self):
        # initialization of a filler
        drying_sphere = mpcd.stream.drying_droplet(R=self.R1, density = self.density, period=1, boundary="no_slip", seed=1991)
        drying_sphere.set_filler(density=5.0, kT=1.0, seed=79, type='A')
        self.assertTrue(drying_sphere._filler is not None)

        mpcd.collide.srd(seed=1221, angle=130., period=1, kT=1.0)

        # run should be able to setup the filler, although this all happens silently
        hoomd.run(2)

        # changing filler should be allowed
        drying_sphere.set_filler(density=10.0, kT=1.5, seed=51)
        self.assertTrue(drying_sphere._filler is not None)
        hoomd.run(1)

        # assert an error is raised if we set a bad particle type
        with self.assertRaises(RuntimeError):
            drying_sphere.set_filler(density=5., kT=1.0, seed=42, type='B')

        # assert an error is raised if we set a bad density
        with self.assertRaises(RuntimeError):
            drying_sphere.set_filler(density=-1.0, kT=1.0, seed=42)

        # removing the filler should still allow a run
        drying_sphere.remove_filler()
        self.assertTrue(drying_sphere._filler is None)
        hoomd.run(1)

    def tearDown(self):
        del self.mpcd_sys

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
