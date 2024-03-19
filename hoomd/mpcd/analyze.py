# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import hoomd
from hoomd.md import _md
from . import _mpcd

import numpy

class radial_solvent_velocity(hoomd.analyze._analyzer):
    R"""Radial solvent velocity analyzer.
    
    Args:
        R (float): Radius for calculation.
        bin_width (float): Bin width for histogramming.
        period (int): Record velocity every *period* time steps.
        phase (int): When -1, start on the current time step. Otherwise, execute
            on steps where *(step + phase) % period* is zero.

    """
    def __init__(self, R, bin_width, period, phase=0):
        hoomd.util.print_status_line()

        if R <= 0.0:
            hoomd.context.msg.error('mpcd.analyze: Cutoff radius must be positive\n')
            raise ValueError('Radius must be positive')

        if bin_width <= 0.0:
            hoomd.context.msg.error('mpcd.analyze: Bin width must be positive\n')
            raise ValueError('Bin width must be positive')

        super(radial_solvent_velocity, self).__init__()
        cpp_class = _mpcd.RadialSolventVelocityAnalyzer
        self.cpp_analyzer = cpp_class(hoomd.context.current.mpcd.data,
                                      R,
                                      bin_width)
        self.setupAnalyzer(period, phase)

        # log metadata fields
        self.metadata_fields = ['R','bin_width','period','phase']
        self.R = R
        self.bin_width = bin_width
        self.period = period
        self.phase = phase

    @property
    def bins(self):
        return numpy.array(self.cpp_analyzer.bins)

    @property
    def density(self):
        num_samples = self.cpp_analyzer.num_samples
        if num_samples > 0:
            return numpy.vstack([self.cpp_analyzer.getDensity(i) for i in range(num_samples)])
        else:
            return None

    @property
    def radial_velocity(self):
        num_samples = self.cpp_analyzer.num_samples
        if num_samples > 0:
            return numpy.vstack([self.cpp_analyzer.getRadialVelocity(i) for i in range(num_samples)])
        else:
            return None

    def reset(self):
        self.cpp_analyzer.reset()