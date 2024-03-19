// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RadialSolventVelocityAnalyzer.cc
 * \brief Declaration of mpcd::RadialSolventVelocityAnalyzer
 */

#include "RadialSolventVelocityAnalyzer.h"
#include "hoomd/extern/pybind/include/pybind11/stl.h"

namespace mpcd
{

/*!
 * \param sysdata MPCD system data
 * \param R Cutoff radius for calculating Radial Density profile
 * \param bin_width Width for binning particle distances
 */
RadialSolventVelocityAnalyzer::RadialSolventVelocityAnalyzer(std::shared_ptr<mpcd::SystemData> sysdata,
                                                             Scalar R,
                                                             Scalar bin_width)
    : Analyzer(sysdata->getSystemDefinition()),
      m_mpcd_sys(sysdata),
      m_mpcd_pdata(m_mpcd_sys->getParticleData()),
      m_R(R)
    {
    m_exec_conf->msg->notice(5) << "Constructing RadialSolventVelocityAnalyzer" << std::endl;

    assert(m_bin_width > 0.0);
    m_num_bins = std::round(m_R / bin_width);
    m_bin_width = m_R / m_num_bins;
    }

RadialSolventVelocityAnalyzer::~RadialSolventVelocityAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying RadialSolventVelocityAnalyzer" << std::endl;
    }

/*!
 * \param timestep Current simulation timestep
 */
void RadialSolventVelocityAnalyzer::analyze(unsigned int timestep)
    {
    // binParticles
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);
    std::vector<unsigned int> counts(m_num_bins, 0);
    std::vector<Scalar> radial_vel(m_num_bins, 0);

    // compute the total number of mpcd particles
    const unsigned int N = m_mpcd_pdata->getN();
    const Scalar Rsq = m_R * m_R;
    for (unsigned int i=0; i < N; ++i)
        {
        const Scalar4 pos_i = h_pos.data[i];
        const Scalar3 r_i = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const Scalar r_isq = dot(r_i, r_i);
        if (r_isq < Rsq)
            {
            const Scalar4 vel_i = h_vel.data[i];
            const Scalar3 v_i = make_scalar3(vel_i.x, vel_i.y, vel_i.z);
            const Scalar r_i_norm = slow::sqrt(r_isq);

            // compute radial velocity, handling special case where r = 0
            Scalar v_irad = 0;
            if (r_i_norm > 0)
                {
                v_irad = dot(v_i, r_i)/r_i_norm;
                }

            const unsigned int bin = static_cast<unsigned int>(r_i_norm / m_bin_width);
            ++counts[bin];
            radial_vel[bin] += v_irad;
            }
        }

#ifdef ENABLE_MPI
    // in MPI, reduce vectors across all ranks so everyone has correct data
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      counts.data(),
                      m_num_bins,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator()
                      );
        MPI_Allreduce(MPI_IN_PLACE,
                      radial_vel.data(),
                      m_num_bins,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator()
                      );
        }
#endif // ENABLE_MPI

    // calculate density and radial velocity from counts
    std::vector<Scalar> density(m_num_bins, 0);
    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        // since density and velocity default to 0, only need to do something
        // if the count is > 0
        if (counts[i] > 0)
            {
            const Scalar r_in = m_bin_width * static_cast<double>(i);
            const Scalar r_out = std::min(r_in + m_bin_width, static_cast<double>(m_R));
            const Scalar V_shell = (4.0*M_PI/3.0) * (r_out * r_out * r_out - r_in * r_in * r_in);
            density[i] = counts[i] / V_shell;
            radial_vel[i] /= counts[i];
            }
        }

    // fill in the elements of density for each frame
    m_density.push_back(density);
    m_radial_vel.push_back(radial_vel);
    }

/*!
 * \returns Bins distribution function was computed on
 */
std::vector<Scalar> RadialSolventVelocityAnalyzer::getBins() const
    {
    std::vector<Scalar> bins(m_num_bins);
    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        const Scalar r_in = m_bin_width * static_cast<Scalar>(i);
        const Scalar r_out = std::min(r_in + m_bin_width, m_R);
        bins[i] = Scalar(0.5) * (r_in + r_out);
        }
    return bins;
    }

unsigned int RadialSolventVelocityAnalyzer::getNumSamples() const
    {
    assert(m_density.size() == m_radial_vel.size());
    return m_density.size();
    }

/*!
 * \returns Radial Density profile at frame idx 
 */
std::vector<Scalar> RadialSolventVelocityAnalyzer::getDensity(unsigned int idx) const
    {
    if (idx >= m_density.size())
        {
        throw std::runtime_error("Frame index to access the m_density is out of range in RadialSolventVelocityAnalyzer");
        }
    return m_density[idx];
    }

/*!
 * \returns Radial velocity profile at frame idx 
 */
std::vector<Scalar> RadialSolventVelocityAnalyzer::getRadialVelocity(unsigned int idx) const
    {
    if (idx >= m_radial_vel.size())
        {
        throw std::runtime_error("Frame index to access the m_radial_velocity is out of range in RadialSolventVelocityAnalyzer");
        }
    return m_radial_vel[idx];
    }

void RadialSolventVelocityAnalyzer::reset()
    {
    m_density.clear();
    m_radial_vel.clear();
    }

namespace detail
{
/*!
 * \param m Python module for export
 */
void export_RadialSolventVelocityAnalyzer(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< RadialSolventVelocityAnalyzer, std::shared_ptr<RadialSolventVelocityAnalyzer> >(m, "RadialSolventVelocityAnalyzer", py::base<Analyzer>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, Scalar, Scalar>())
        .def_property_readonly("num_samples", &RadialSolventVelocityAnalyzer::getNumSamples)
        .def("getDensity", &RadialSolventVelocityAnalyzer::getDensity)
        .def("getRadialVelocity", &RadialSolventVelocityAnalyzer::getRadialVelocity)
        .def_property_readonly("bins", &RadialSolventVelocityAnalyzer::getBins)
        .def("reset", &RadialSolventVelocityAnalyzer::reset);
    }
} // end namespace detail
} // end namespace mpcd

