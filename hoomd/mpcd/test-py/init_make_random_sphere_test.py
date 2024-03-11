# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import hoomd
import numpy as np
from hoomd import mpcd

# unit tests for generating random solvent particles inside sphere
class mpcd_snapshot(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()
        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=120.)))

    def test_init(self):
        density = 5.
        R0 = 30. # radius of sphere
        s = mpcd.init.make_random_sphere(density=density, R=R0, kT =1.0, seed=7)
        snap = s.take_snapshot()

        if hoomd.comm.get_rank() == 0:
            r = np.linalg.norm(snap.particles.position, axis=1)

            # all particles must be inside the sphere
            self.assertTrue(np.all(r < R0))

            # center of mass velocity should be zero
            vcm = np.mean(snap.particles.velocity, axis=0)
            self.assertTrue(np.allclose(vcm, [0, 0, 0]))

            # particles should have roughly the right average density
            self.assertAlmostEqual(snap.particles.N/(4*np.pi*R0**3/3), density, delta=0.5)

            # particles should be uniformly distributed in r
            hist,bin_edges = np.histogram(r, bins=int(R0), range=(0, R0))
            bin_volumes = (4.*np.pi/3.)*(bin_edges[1:]**3 - bin_edges[:-1]**3)
            bin_density = hist / bin_volumes
            self.assertTrue(np.allclose(bin_density, density, rtol=0.6))

            # cos(phi) should be uniformly distributed
            cos_phi = snap.particles.position[:, 2]/r
            hist_cos_phi, _ = np.histogram(cos_phi, bins=20, density=True)
            self.assertTrue(np.allclose(hist_cos_phi , 0.5, rtol=0.1))

            # theta should be uniformly distributed
            theta = np.arctan2(snap.particles.position[:, 1], snap.particles.position[:, 0])
            hist_theta, _ = np.histogram(theta, bins=20, density=True)
            self.assertTrue(np.allclose(hist_theta , 1/(2*np.pi), rtol=0.1))

    # test that make_random_sphere is also working fine in 2D
    def test_init_2D(self):
        # clear out the system
        hoomd.context.initialize()
        # reinitialize the system with box having dimensions = 2
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=120., dimensions=2)))
        density = 5.
        R0 = 30. # radius of sphere
        s = mpcd.init.make_random_sphere(density=density, R=R0, kT =1.0, seed=7)
        snap = s.take_snapshot()

        if hoomd.comm.get_rank() == 0:
            r = np.linalg.norm(snap.particles.position, axis=1)

            # center of mass velocity should be zero
            vcm = np.mean(snap.particles.velocity, axis=0)
            self.assertTrue(np.allclose(vcm, [0, 0, 0]))

            # z components should be zero in 2d
            self.assertTrue(np.allclose(snap.particles.position[:, 2], 0))
            self.assertTrue(np.allclose(snap.particles.velocity[:, 2], 0))

            # all particles must be inside the sphere
            self.assertTrue(np.all(r < R0))

            # particles should have roughly the right average density
            self.assertAlmostEqual(snap.particles.N/(np.pi*R0**2), density, delta=0.5)

            # particles should be uniformly distributed in r
            hist,bin_edges = np.histogram(r, bins=int(R0), range=(0, R0))
            bin_volumes = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
            bin_density = hist / bin_volumes
            self.assertTrue(np.allclose(bin_density, density, rtol=0.6))

            # theta should be uniformly distributed
            theta = np.arctan2(snap.particles.position[:, 1], snap.particles.position[:, 0])
            hist_theta, _ = np.histogram(theta, bins=20, density=True)
            self.assertTrue(np.allclose(hist_theta , 1/(2*np.pi), rtol=0.1))

    # test that if the radius of sphere 0 or negative raise an error 
    def test_negative_radius(self):
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=0.0, kT =1.0, seed=7)
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=-5.0, kT =1.0, seed=7)

    # diameter of sphere larger than length of box should raise an error 
    def test_radius_too_big(self):
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=70.0, kT =1.0, seed=7)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

