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

    // allocate memory for the bins
    m_num_bins = std::round(m_R / bin_width);
    m_bin_width = m_R / m_num_bins;
    reset();

#ifdef ENABLE_MPI
    // MPI is currently not supported due to how this analyzer might degrade performance
    if (m_exec_conf->getNRanks() > 1)
        {
        m_exec_conf->msg->error() << "MPCD: Radial Solvent Velocity analyzer does not support MPI" << std::endl;
        throw std::runtime_error("Radial Solvent Velocity analyzer does not support MPI");
        }
#endif // ENABLE_MPI
    }

RadialSolventVelocityAnalyzer::~RadialSolventVelocityAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying RadialSolventVelocityAnalyzer" << std::endl;
    }

/*!
 * \param timestep Current simulation timestep
 *
 * To get rho(r), we calculate as
 *
 * \verbatim
 *                  counts_i
 *      g_i = ----------------------
 *                  V_shell
 *
 * \endverbatim
 *
 * The shell volume is V_shell = (4 pi/3)*(r_{i+1}^3-r_i^3).
 */
void RadialSolventVelocityAnalyzer::analyze(unsigned int timestep)
    {
    if (m_prof) m_prof->push("RadialSolventVelocityAnalayse");

    // binParticles
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);
    std::vector<unsigned int> counts(m_num_bins, 0);
    std::vector<Scalar> radial_vel(m_num_bins, 0);

    // compute the total number of mpcd particles
    const unsigned int N = m_mpcd_pdata->getN();
    for (unsigned int i=0; i < N; ++i)
        {
        // load particle i
        const Scalar4 pos_i = h_pos.data[i];
        const Scalar4 vel_i = h_vel.data[i];
        const Scalar3 r_i = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        const Scalar3 v_i = make_scalar3(vel_i.x, vel_i.y, vel_i.z);
        
        // distance and radial velocity calculation 
        const Scalar r_isq = dot(r_i, r_i);
        const Scalar r_i_norm = slow::sqrt(r_isq);
        const Scalar v_irad = dot(v_i, r_i)/r_i_norm;

        if (r_i_norm < m_R)
            {
            const unsigned int bin = static_cast<unsigned int>(r_i_norm / m_bin_width);
            ++counts[bin];
            radial_vel[bin] += v_irad;
            }
        }

    // calculate density and radial velocity from counts
    std::vector<Scalar> density(m_num_bins, 0);
    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        const double r_in = m_bin_width * static_cast<double>(i);
        const double r_out = std::min(r_in + m_bin_width, static_cast<double>(m_R));
        const double V_shell = (4.0*M_PI/3.0) * (r_out * r_out * r_out - r_in * r_in * r_in);
        density.push_back(counts[i]/V_shell);
        }

    // fill in the elements of density for each frame
    m_density.push_back(density);
    m_radial_vel.push_back(radial_vel);

    if (m_prof) m_prof->pop();
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

/*!
 * \returns Radial Density profile at frame idx 
 */
std::vector<Scalar> RadialSolventVelocityAnalyzer::get(unsigned int idx) const
    {
    if (idx > m_density.size())
        {
        throw std::runtime_error("Frame index to access the m_density is out of range in RadialSolventVelocityAnalyzer");
        }
    std::vector<Scalar> density;
    density = m_density[idx];
    return density;
    }

/*!
 * \returns Radial velocity profile at frame idx 
 */
std::vector<Scalar> RadialSolventVelocityAnalyzer::getRadialVelocity(unsigned int idx) const
    {
    if (idx > m_radial_vel.size())
        {
        throw std::runtime_error("Frame index to access the m_radial_velocity is out of range in RadialSolventVelocityAnalyzer");
        }
    std::vector<Scalar> rad_vel;
    rad_vel = m_radial_vel[idx];
    return rad_vel;
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
        .def("get", &RadialSolventVelocityAnalyzer::get)
        .def("getRadialVelocity", &RadialSolventVelocityAnalyzer::getRadialVelocity)
        .def("getBins", &RadialSolventVelocityAnalyzer::getBins)
        .def("reset", &RadialSolventVelocityAnalyzer::reset);
    }
} // end namespace detail
} // end namespace mpcd

