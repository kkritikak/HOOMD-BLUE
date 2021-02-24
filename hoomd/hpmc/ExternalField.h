// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _EXTERNAL_FIELD_H_
#define _EXTERNAL_FIELD_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/

#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"

#include "HPMCCounters.h" // do we need this to keep track of the statistics?

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hpmc
    {
class ExternalField : public Compute
    {
    public:		
    /// Compute the external potential energy of a given system.
    virtual double computeSystemEnergy()
        {
        return 0;
        }
    ExternalField(std::shared_ptr<SystemDefinition> sysdef) : Compute(sysdef) { }
    };
//! Compute that accepts or rejects moves according to some external field
/*! **Overview** <br>

    \ingroup hpmc_computes
*/
template<class Shape> class ExternalFieldMono : public ExternalField
    {
    public:
    ExternalFieldMono(std::shared_ptr<SystemDefinition> sysdef) : ExternalField(sysdef) { }

    ~ExternalFieldMono() { }

    //! method to calculate the potential energy of a single particle.
    virtual double
    evaluateEnergy(const unsigned int index, const vec3<Scalar>& position, const Shape& shape)
        {
        return 0;
        }
    };

template<class Shape> void export_ExternalFieldMono(pybind11::module& m, std::string name)
    {
    pybind11::class_<ExternalFieldMono<Shape>,
                     ExternalField,
                     std::shared_ptr<ExternalFieldMono<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

void export_ExternalField(pybind11::module& m)
    {
    pybind11::class_<ExternalField, Compute, std::shared_ptr<ExternalField>>(
        m,
        "ExternalField")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("computeSystemEnergy", &ExternalField::computeSystemEnergy);
    }

    } // end namespace hpmc

#endif // end inclusion guard
