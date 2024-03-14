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
 * \param bin_width Width for binning particle distances
 */
RadialSolventVelocityAnalyzer::RadialSolventVelocityAnalyzer(std::shared_ptr<mpcd::SystemData> sysdata,
                                                             Scalar bin_width)
    : m_mpcd_sys(sysdata),
      Analyzer(m_mpcd_sys->getSystemDefinition()),
      m_mpcd_pdata(m_mpcd_sys->getParticleData()),
      m_sysdef(m_mpcd_sys->getSystemDefinition()),
      m_global_box(m_sysdef->getParticleData()->getGlobalBox()),
      m_bin_width(bin_width),
      m_Rmax(0.0),
      m_num_samples(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing RadialSolventVelocityAnalyzer" << std::endl;

    assert(m_bin_width > 0.0);

    // allocate memory for the bins
    Scalar3 L = m_global_box.getL();
    Scalar m_Rmax = std::min( L.x, std::min( L.y, L.z));
    m_num_bins = std::ceil(m_Rmax / m_bin_width);
    GPUArray<unsigned int> counts(m_num_bins, m_exec_conf);
    m_counts.swap(counts);
    GPUArray<double> accum_rho(m_num_samples, m_num_bins, m_exec_conf);
    m_accum_rho.swap(accum_rho);
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
 */
void RadialSolventVelocityAnalyzer::analyze(unsigned int timestep)
    {
    if (m_prof) m_prof->push("RadialSolventVelocityAnalayse");

    binParticles();

    accumulate();

    if (m_prof) m_prof->pop();
    }

/*!
 * Particle pairs are binned into *m_counts* 
 */
void RadialSolventVelocityAnalyzer::binParticles()
    {
    if (m_prof) m_prof->push("bin");

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_counts(m_counts, access_location::host, access_mode::overwrite);
    memset(h_counts.data, 0, m_num_bins * sizeof(unsigned int));

    const Scalar Rsq = m_Rmax * m_Rmax;

    // compute the total number of mpcd particles
    const unsigned int N = m_mpcd_pdata->getN();
    for (unsigned int i=0; i < N; ++i)
        {
        // load particle i
        const Scalar4 pos_i = h_pos.data[i];
        const Scalar3 r_i = make_scalar3(pos_i.x, pos_i.y, pos_i.z);

        // distance calculation
        const Scalar r_isq = dot(r_i, r_i);

        if (r_isq < Rsq)
            {
            const unsigned int bin = static_cast<unsigned int>(slow::sqrt(r_isq) / m_bin_width);
            ++h_counts.data[bin];
            }
        }

    if (m_prof) m_prof->pop();
    }

/*!
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
void RadialSolventVelocityAnalyzer::accumulate()
    {
    if (m_prof) m_prof->push("accumulate");

    if (m_num_samples > 0)
        {
        m_accum_rho.resize(m_num_samples, m_num_bins);
        }

    ArrayHandle<unsigned int> h_counts(m_counts, access_location::host, access_mode::read);
    ArrayHandle<double> h_accum_rho(m_accum_rho, access_location::host, access_mode::overwrite);

    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        const double r_in = m_bin_width * static_cast<double>(i);
        const double r_out = std::min(r_in + m_bin_width, static_cast<double>(m_Rmax));
        const double V_shell = (4.0*M_PI/3.0) * (r_out * r_out * r_out - r_in * r_in * r_in);
        h_accum_rho.data[i + m_num_samples] += static_cast<double>(h_counts.data[i])/ V_shell;
        }
    ++m_num_samples;
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
        const Scalar r_out = std::min(r_in + m_bin_width, m_Rmax);
        bins[i] = Scalar(0.5) * (r_in + r_out);
        }
    return bins;
    }

/*!
 * \returns Accumulated density
 */
std::vector<std::vector<Scalar>> RadialSolventVelocityAnalyzer::get() const
    {
    ArrayHandle<double> h_accum_rho(m_accum_rho, access_location::host, access_mode::read);
    std::vector<std::vector<Scalar>> rho(m_num_samples, std::vector<Scalar>(m_num_bins));
    if (m_num_samples > 0)
        {
        for (unsigned int i=0; i < m_num_bins; ++i)
            {
            for (unsigned int j=0; j < m_num_samples; ++j)
                {
                rho[j][i] = h_accum_rho.data[i + j];
                }            
            }
        }
    return rho;
    }

void RadialSolventVelocityAnalyzer::reset()
    {
    m_num_samples = 0;
    ArrayHandle<double> h_accum_rho(m_accum_rho, access_location::host, access_mode::overwrite);
    memset(h_accum_rho.data, 0, m_num_bins * sizeof(double));
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
        .def(py::init<std::shared_ptr<mpcd::SystemData>, Scalar>())
        .def("get", &RadialSolventVelocityAnalyzer::get)
        .def("getBins", &RadialSolventVelocityAnalyzer::getBins)
        .def("reset", &RadialSolventVelocityAnalyzer::reset);
    }
} // end namespace detail
} // end namespace mpcd