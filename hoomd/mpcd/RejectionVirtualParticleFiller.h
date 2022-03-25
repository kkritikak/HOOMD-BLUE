
#ifndef MPCD_REJECTION_FILLER_H_
#define MPCD_REJECTION_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to MPCD particle data for "sphere" geometry
/*!
 * <detailed description needed>
*/

class PYBIND11_EXPORT RejectionVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
        RejectionVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                        Scalar density,
                        unsigned int type,
                        std::shared_ptr<::Variant> T,
                        unsigned int seed);

        virtual ~RejectionVirtualParticleFiller();

    protected:
        //! Fill the particles in the padding region
        virtual void fill();
    };

namespace detail
{
//! Export RejectionVirtualParticleFiller to python
void export_RejectionVirtualParticleFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_FILLER_H_
