# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import mpcd

class mpcd_analyze_radial_solvent_velocity_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=3)
        snap.particles.position[:] = [[0, 0, 0], [0.1, 0, 0], [0, 1.1, 0]]
        snap.particles.velocity[:] = [[1, 1, 1], [2, 1, -1.], [1, 2, 3]]
        self.s = mpcd.init.read_snapshot(snap)

        # create an integrator
        mpcd.integrator(dt=0.0)

    def test_create(self):
        mpcd.analyze.radial_solvent_velocity(R=5, bin_width=0.5, period=1)

    def test_calculate(self):
        # check initialization
        vel = mpcd.analyze.radial_solvent_velocity(R=5, bin_width=0.5, period=2)
        bin_edges = np.linspace(0, 5, 11)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_volumes = (4*np.pi/3) * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        np.testing.assert_allclose(vel.bins, bin_centers)
        self.assertIsNone(vel.density)
        self.assertIsNone(vel.radial_velocity)

        # this should trigger a calculation
        hoomd.run(1)
        # check density
        ref_density = np.zeros(10)
        ref_density[0] = 2 / bin_volumes[0]
        ref_density[2] = 1 / bin_volumes[2]
        np.testing.assert_allclose(vel.density.shape, (1, 10))
        np.testing.assert_allclose(vel.density[0], ref_density)
        # check velocity
        ref_velocity = np.zeros(10)
        ref_velocity[0] = 1
        ref_velocity[2] = 2
        np.testing.assert_allclose(vel.radial_velocity.shape, (1, 10))
        np.testing.assert_allclose(vel.radial_velocity[0], ref_velocity)

        # nothing should happen on this step
        hoomd.run(1)
        np.testing.assert_allclose(vel.density.shape, (1, 10))
        np.testing.assert_allclose(vel.radial_velocity.shape, (1, 10))

        # now run one more and make sure we get another profile
        hoomd.run(1)
        np.testing.assert_allclose(vel.density.shape, (2, 10))
        np.testing.assert_allclose(vel.density[0], ref_density)
        np.testing.assert_allclose(vel.density[1], ref_density)
        np.testing.assert_allclose(vel.radial_velocity.shape, (2, 10))
        np.testing.assert_allclose(vel.radial_velocity[0], ref_velocity)
        np.testing.assert_allclose(vel.radial_velocity[1], ref_velocity)

        # take a snapshot, change some stuff, then make sure new profiles don't
        # match the old ones
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0, 2] = 1.2
            snap.particles.velocity[0, 2] = -6
        self.s.restore_snapshot(snap)
        hoomd.run(2)
        np.testing.assert_allclose(vel.density.shape, (3, 10))
        np.testing.assert_allclose(vel.density[0], ref_density)
        np.testing.assert_allclose(vel.density[1], ref_density)
        ref_density[0] = 1 / bin_volumes[0]
        ref_density[2] = 2 / bin_volumes[2]
        np.testing.assert_allclose(vel.density[2], ref_density)
        np.testing.assert_allclose(vel.radial_velocity.shape, (3, 10))
        np.testing.assert_allclose(vel.radial_velocity[0], ref_velocity)
        np.testing.assert_allclose(vel.radial_velocity[1], ref_velocity)
        ref_velocity[0] = 2
        ref_velocity[2] = -2
        np.testing.assert_allclose(vel.radial_velocity[2], ref_velocity)

        # reset and make sure everything clears
        vel.reset()
        self.assertIsNone(vel.density)
        self.assertIsNone(vel.radial_velocity)

    def test_R_zero(self):
        with self.assertRaises(ValueError):
            mpcd.analyze.radial_solvent_velocity(R=0, bin_width=0.1, period=1)

    def test_R_negative(self):
        with self.assertRaises(ValueError):
            mpcd.analyze.radial_solvent_velocity(R=-1, bin_width=0.1, period=1)

    def test_bin_width_zero(self):
        with self.assertRaises(ValueError):
            mpcd.analyze.radial_solvent_velocity(R=5, bin_width=0, period=1)

    def test_bin_width_negative(self):
        with self.assertRaises(ValueError):
            mpcd.analyze.radial_solvent_velocity(R=5, bin_width=-0.1, period=1)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
