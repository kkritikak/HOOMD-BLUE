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
        message = "Global number of particles inside droplet is increasing after decreasing the radius"
        if hoomd.comm.get_rank() == 0:
            num_par_After = snap.particles.N
            self.assertLessEqual(num_par_After, self.num_particles, message)

        # take another step and see if number of particles are decreasing again
        hoomd.run(1)
        snap2 = self.mpcd_sys.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            num_par_After = snap2.particles.N
            self.assertLessEqual(num_par_After, self.num_particles, message)

    def tearDown(self):
        del self.mpcd_sys

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
