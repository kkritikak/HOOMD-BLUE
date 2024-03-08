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

    def test_set_density(self):
        density = 5.
        R0 = 30. # radius of sphere
        s = mpcd.init.make_random_sphere(density=density, R=R0, kT =1.0, seed=7)
        snap = s.take_snapshot()
        # calculating the average density and radial density profile and checking it's equal to density
        dr = 1.0 # shell thickness to calculate density
        nshell = int(np.round(R0/dr)) 
        R = np.zeros(nshell) 
        Vshell = np.zeros(nshell) 
        particles_inshell = np.zeros(nshell)
        if hoomd.comm.get_rank() == 0:
            # calculating average density 
            avg_dens = snap.particles.N / ((4./3.)*np.pi*R0**3)
            self.assertAlmostEqual(avg_dens, density, delta=0.5)

            for h in range(nshell):
                R[h]=(h+1)*dr
                Vshell[h]=(4.0/3.0)*np.pi*(((h+1)*dr)**3 - (h*dr)**3)
            for i in range(snap.particles.N):
                posi = snap.particles.position[i]
                posi_mod = np.linalg.norm(posi)
                self.assertTrue(posi_mod < R0)
                shell = int(posi_mod/dr)
                particles_inshell[shell] += 1
            density_array = np.array(particles_inshell/Vshell)

            self.assertTrue(np.allclose(density_array, density, atol=0.6))

    # test that if the radius of sphere 0 or negative raises an error
    def test_negative_radius(self):
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=0.0, kT =1.0, seed=7)
        with self.assertRaises(RuntimeError):
            s = mpcd.init.make_random_sphere(density=5.0, R=-5.0, kT =1.0, seed=7)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])