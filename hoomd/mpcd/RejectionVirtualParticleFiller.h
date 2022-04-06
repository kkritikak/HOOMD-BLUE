
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

template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
        //! Constructor
        /*!
         * \param sysdata - MPCD system data
         * \param density - number density of MPCD solvent
         * \param type - particle fill type
         * \param T - Temperature
         * \param seed - seed for PRNG
         * \param geom - confinement geometry
         */
        RejectionVirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                       Scalar density,
                                       unsigned int type,
                                       std::shared_ptr<::Variant> T,
                                       unsigned int seed,
                                       std::shared_ptr<const Geometry> geom);

        //! Destructor
        virtual ~RejectionVirtualParticleFiller();

        //! Get the streaming geometry
        std::shared_ptr<const Geometry> getGeometry() const
            {
            return m_geom;
            }

        //! Set the streaming geometry
        void setGeometry(std::shared_ptr<const Geometry> geom)
            {
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const Geometry> m_geom;

        //! Fill the particles outside the confinement
        void fill(unsigned int timestep);
    };

namespace detail
{
//! Export RejectionVirtualParticleFiller to python
template<class Geometry>
void export_RejectionVirtualParticleFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_REJECTION_FILLER_H_
