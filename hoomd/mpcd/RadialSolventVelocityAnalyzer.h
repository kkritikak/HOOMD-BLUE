// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RadialSolventVelocityAnalyzer.h
 * \brief Declaration of mpcd::RadialSolventVelocityAnalyzer
 */

#ifndef MPCD_RADIAL__SOLVENT_VELOCITY_ANALYZER_H_
#define MPCD_RADIAL_SOLVENT_VELOCITY_ANALYZER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Analyzer.h"
#include "hoomd/GPUFlags.h"
#include "SystemData.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Solvent Density analyzer
/*!
 * The RadialSolventVelocityAnalyzer computes the Radial Solvent density during the simulation and Average radial velocity profile.
 * The total density is accumulated internally over multiple frames so we can get it at multiple times.
 *
 */

class PYBIND11_EXPORT RadialSolventVelocityAnalyzer : public Analyzer
    {
    public:
        //! Constructor
        RadialSolventVelocityAnalyzer(std::shared_ptr<mpcd::SystemData> sysdata,
                                      Scalar bin_width);

        //! Destructor
        virtual ~RadialSolventVelocityAnalyzer();

        //! Perform radial solvent density analysis
        virtual void analyze(unsigned int timestep);

        //! Get a copy of the bins
        std::vector<Scalar> getBins() const;

        //! Get a copy of the density data
        std::vector<std::vector<Scalar>> get() const;

        //! Reset the accumulated values
        void reset();

    protected:
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;   //!< MPCD Particle data
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;       //!< MPCD System
        const BoxDim& m_global_box;                         //!< Global simulation box 
        std::shared_ptr<SystemDefinition> m_sysdef;

        GPUArray<unsigned int> m_counts;    //!< Array of bin counts from a single frame
        Scalar m_Rmax;                         //!< Maximum radius possible for a sphere inside box
        Scalar m_bin_width;                 //!< Bin width
        unsigned int m_num_bins;            //!< Number of bins
        unsigned int m_num_samples;         //!< Number of samples where analysis is done 

        GPUArray<double> m_accum_rho;     //!< Accumulated density

        //! Bin particle counts
        virtual void binParticles();

        //! Accumulate the counts into the density profile
        virtual void accumulate();
    };

namespace detail
{
//! Exports the RadialSolventVelocityAnalyzer to python
void export_RadialSolventVelocityAnalyzer(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_RADIAL_SOLVENT_VELOCITY_ANALYZER_H_