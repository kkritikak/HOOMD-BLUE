
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
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=105.)))

        # initialize the system from the starting snapshot, intial solvent density 0.10
        density = 5.
        R = 50.
        Pe = 5.
        dt = 0.001
        alpha = 8*np.pi*Pe
        print("alpha" , alpha)
        num_particles = int(density*(4./3.)*np.pi*R*R*R)

        print("number of particles in the droplet",num_particles)
        #making snapshot
        self.mpcd_sys = hoomd.mpcd.init.make_random(N=num_particles, kT=1.0, seed=np.random.randint(1000))
        snap = self.mpcd_sys.take_snapshot()

        # modify snapshot particles positions so they lie inside droplet


        particles = []
        while len(particles) < num_particles:
            x = np.random.uniform(-R, R)
            y = np.random.uniform(-R, R)
            z = np.random.uniform(-R, R)
            if x**2 + y**2 + z**2 <= R**2:
                particles.append((x, y, z))



        snap.particles.position[:] = particles[:]
        #restoring the snapshot
        self.mpcd_sys.restore_snapshot(snap)
        mpcd.integrator(dt=dt)


    def test_for_N_bounced(self):
        R = 50.
        density = 5.
        Pe = 5.
        alpha = 8.*np.pi*Pe
        dt = 0.001
        #creating a variant for R
        def variant_R(t):
            R_t = np.sqrt(R*R - (alpha*t*dt/(4.*np.pi)))
            return R_t
        
        my_variant = []
        for i in range(1600):
            my_variant.append((i , variant_R(i)))

        R1 = hoomd.variant.linear_interp(points = my_variant)
        self.mpcd_sys.sorter.set_period(period = 100)                                 
        mpcd.stream.sphere(R=R1, density = density,period=10, boundary="no_slip")
        mpcd.collide.srd(seed=1991, period=10, angle=130., kT=1.0, group=hoomd.group.all())

        # take 20 step
        hoomd.run(1500)
        
if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
