
import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd
from mpi4py import MPI

# unit tests for mpcd sphere streaming geometry
class mpcd_stream_sphere_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()
        hoomd.option.set_notice_level(5)
        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=105.)))

        # initialize the system from the starting snapshot
        density = 5.
        R = 50.
        Pe = 5.
        dt = 0.001
        alpha = 8*np.pi*Pe
        num_particles = int(density*(4./3.)*np.pi*R*R*R)
        #making snapshot
        self.mpcd_sys = hoomd.mpcd.init.make_random(N=num_particles, kT=1.0, seed=np.random.randint(1000))
        snap = self.mpcd_sys.take_snapshot()

        # modify snapshot particles positions so they lie inside droplet
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            particles = []
            while len(particles) < (snap.particles.N):
                x = np.random.uniform(-R, R)
                y = np.random.uniform(-R, R)
                z = np.random.uniform(-R, R)
                if x**2 + y**2 + z**2 <= R**2:
                    particles.append((x, y, z))
            snap.particles.position[:] = particles[:]

        #restoring the snapshot
        self.mpcd_sys.restore_snapshot(snap)

        mpcd.integrator(dt=dt)

    def test_for_globalN(self):
        R = 50.
        density = 5.
        Pe = 5.
        alpha = 8.*np.pi*Pe
        dt = 0.001
        num_particles = int(density*(4./3.)*np.pi*R*R*R)
        #creating a variant for R
        def variant_R(t):
            R_t = np.sqrt(R*R - (alpha*t*dt/(4.*np.pi)))
            return R_t
        
        my_variant = []
        for i in range(150):
            my_variant.append((i , variant_R(i)))

        R1 = hoomd.variant.linear_interp(points = my_variant)

        #initializing the droplet
        self.mpcd_sys.sorter.set_period(period = 100)            
        drying_sphere = mpcd.stream.drying_droplet(R=R1, density = density,period=10, boundary="no_slip",seed=np.random.randint(1000))

        # take 100 step
        hoomd.run(100)
        snap2 = self.mpcd_sys.take_snapshot()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        message = "Global number of particles inside droplet is increasing after decreasing the radius"
        if rank == 0:
            num_par_After = snap2.particles.N
            self.assertLessEqual(num_par_After, num_particles, message)
if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
