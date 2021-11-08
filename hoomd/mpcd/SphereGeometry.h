
#ifndef MPCD_SPHERE_GEOMETRY_H_
#define MPCD_SPHERE_GEOMETRY_H_

#include <math.h>

#include "BoundaryCondition.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // NVCC

namespace mpcd
{
namespace detail
{
//! Spherical droplet geometry
/*!
 * TODO: write proper description of what is being done in the class below
 */
class __attribute__((visibility("default"))) SphereGeometry
    {
    public:
        //! Constructor
        /*!
         * \param R confinement radius
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SlitGeometry(Scalar R, boundary bc)
            : m_R(R), m_bc(bc)
            { }

        //! Detect collision between the particle and the boundary
        /*!
         * \param pos Proposed particle position
         * \param vel Proposed particle velocity
         * \param dt Integration time remaining
         *
         * \returns True if a collision occurred, and false otherwise
         *
         * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel is updated
         *       according to the appropriate bounce back rule, and the integration time \a dt is decreased to the
         *       amount of time remaining.
         */
        HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
            {

            /*
             * Detect if particle has left the box, and try to avoid branching or absolute value calls. The sign used
             * in calculations is 1 if the particle is out-of-bounds in the radial direction, 0 otherwise.
             *
             * We intentionally use > / < rather than >= / <= to make sure that spurious collisions do not get detected
             * when a particle is reset to the boundary location. A particle landing exactly on the boundary from the bulk
             * can be immediately reflected on the next streaming step, and so the motion is essentially equivalent up to
             * an epsilon of difference in the channel width.
             */
            double r2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z;
            const char sign = (r2 > R*R);
            // exit immediately if no collision is found or particle is not moving normal (radial direction) to the wall
            // (since no new collision could have occurred if there is no normal motion)
            /*
            r(t) = sqrt(x**2 + y**2 + z**2)
            radial velocity, r_dot = (x*x_dot + y*y_dot + z*z_dot)/r(t)
            */
            float vr = (pos.x*vel.x + pos.y*vel.y + pos.z*vel.z)/sqrt(r2);
            if (sign == 0 || vr == Scalar(0))
               {
               dt = Scalar(0);
               return false;
               }

            // Find the point of contact when the particle is just leaving the surface
            /*
            * Calculating the time spent outside-bounds requires the knowledge of the point of contact on the boundary.
            * The point of contact can be calculated using geometrical considerations.
            *
            * We know the following quantities,
            *    1. r(t+del_t)      (the point outside the sphere)
            *    2. v(t)            (previous velocity)
            *
            * Assuming 'r*' is the point of contact on the spherical shell and dx* is the distance travelled by the
            * particle outside the boundary.
            *
            * r* = r(t+del_t) + dx* * v(t)/|v|          (the -ve signs cancel out in the 2nd term)
            * and |r|**2 = R**2
            *
            * Solving the above equations,
            * dx* = -r(t+del_t).v(t)/|v| +- sqrt((r(t+del_t).v(t))**2 - |r(t+del_t)|**2 + R**2)
            * consequently, dt = dx* / |v|
            */
            double mod_v,r_dot_v_hat,dx_;

            mod_v = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
            r_dot_v_hat = (pos.x*vel.x + pos.y*vel.y + pos.z*vel.z)/mod_v;

            dx_ = -r_dot_v_hat + sqrt(r_dot_v_hat*r_dot_v_hat - r2 + R*R);
            dt = dx_/mod_v;

            // backtrack the particle for time dt to get to point of contact
            pos.x -= vel.x*dt;
            pos.y -= vel.y*dt;
            pos.z -= vel.z*dt;

            // update velocity according to boundary conditions
            // no-slip requires reflection of the tangential components
            if (m_bc == boundary::no_slip)
                {
                /*
                Both the tangential and perpendicular components must be reflected.
                */
                vel.x = -vel.x;
                vel.y = -vel.y;
                vel.z = -vel.z;
                }

            return true;
            }

        //! Check if a particle is out of bounds
        /*!
         * \param pos Current particle position
         * \returns True if particle is out of bounds, and false otherwise
         */
        HOSTDEVICE bool isOutside(const Scalar3& pos) const
            {
            double r2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z;
            return (r2 > R*R);
            }

        //! Validate that the simulation box is large enough for the geometry
        /*!
         * \param box Global simulation box
         * \param cell_size Size of MPCD cell
         *
         * The box is large enough if the shell is padded along the radial direction, so that cells at the boundary
         * would not interact with each other via PBC.
         *
         * It would be enough to check the padding along the x,y,z directions individually as the box boundaries are
         * closest to the sphere boundary along these axes.
         */
        HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
            {
            const Scalar3 hi;
            const Scalar3 lo;
            hi.x = box.getHi().x; hi.y = box.getHi().y; hi.z = box.getHi().z;
            lo.x = box.getLo().x; lo.y = box.getLo().y; lo.z = box.getLo().z;

            return ((hi.x-m_R) >= cell_size && (-lo.x-m_R) >= cell_size &&
                    (hi.y-m_R) >= cell_size && (-lo.y-m_R) >= cell_size &&
                    (hi.y-m_R) >= cell_size && (-lo.y-m_R) >= cell_size );
            }

        //! Get Sphere radius
        /*!
         * \returns confinement radius
         */
        HOSTDEVICE Scalar getR() const
            {
            return m_R;
            }

        //! Get the wall boundary condition
        /*!
         * \returns Boundary condition at wall
         */
        HOSTDEVICE boundary getBoundaryCondition() const
            {
            return m_bc;
            }

        #ifndef NVCC
        //! Get the unique name of this geometry
        static std::string getName()
            {
            return std::string("Sphere");
            }
        #endif // NVCC

    private:
        const Scalar m_R;       //!< Spherical confinement radius
        const boundary m_bc;    //!< Boundary condition
    };

} // end namespace detail
} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_SPHERE_GEOMETRY_H_
