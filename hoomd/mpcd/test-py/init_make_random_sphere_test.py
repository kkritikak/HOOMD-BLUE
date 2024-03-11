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

            # particles should have roughly the right average density
            self.assertAlmostEqual(snap.particles.N/(4*np.pi*R0**3/3), density, delta=0.5)

            # particles should be uniformly distributed in r
            hist,bin_edges = np.histogram(r, bins=int(R0), range=(0, R0))
            bin_volumes = (4.*np.pi/3.)*(bin_edges[1:]**3 - bin_edges[:-1]**3)
            bin_density = hist / bin_volumes
            self.assertTrue(np.allclose(bin_density, density, rtol=0.6))

            # cos(phi) should be uniformly distributed
            cos_phi = snap.particles.position[:, 2]/r
            hist_cos_phi, bins_cos_phi = np.histogram(cos_phi, bins=20)
            self.assertTrue(np.allclose(hist_cos_phi , np.mean(hist_cos_phi), rtol=0.1))

            # theta should be uniformly distributed
            theta = np.arctan2(snap.particles.position[:, 1], snap.particles.position[:, 0])
            hist_theta, bins_theta = np.histogram(theta, bins=20)
            self.assertTrue(np.allclose(hist_theta , np.mean(hist_theta), rtol=0.1))

    # test that if the radius of sphere 0 or negative or greater than boxdim raises an error
    def test_negative_radius(self):
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=0.0, kT =1.0, seed=7)
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=-5.0, kT =1.0, seed=7)
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=70.0, kT =1.0, seed=7)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

