
#ifndef MPCD_SPHERE_GEOMETRY_H_
#define MPCD_SPHERE_GEOMETRY_H_

#include "BoundaryCondition.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#include <cmath>

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
//! Sphere geometry
/*!
 * This models a fluid confined inside a sphere, centered at the origin and with initial radius R.
 *
 * If a particle leaves the sphere in a single simulation step, the particle is backtracked to the point on the
 * surface from which it exited the surface and then reflected according to appropriate boundary condition.
 */
class __attribute__((visibility("default"))) SphereGeometryRetracting
    {
    public:
        //! Constructor
        /*!
         * \param R confinement radius initially, which is defined as m_R0
         * \param bc Boundary condition at the wall (slip or no-slip)
	 * \param V Velocity of spherical container
	 * \param t is time
	 * \\m_R is the radius of container at time t, m_R0( intial radius) = m_R( radius at that time step) - V*(t-t'), t' is the initial time(taking it as zero)
	 * \so m_R0 = m_R -V*t and m_R = m_R0(initial R) + V*t
	 * \m_R2 is m_R*m_R
         */
        HOSTDEVICE SphereGeometry(Scalar R, Scalar V, Scalar t, boundary bc)
            : m_R(R+V*t), m_R2((R+V*t)*(R+V*t)), m_bc(bc), m_V(V), m_V2(V*V), m_R0(R)
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
             * If particle is still inside the sphere or has zero speed, no collision could have occurred and therefore
             * exit immediately.
             */
            const Scalar r2 = dot(pos,pos);
            const Scalar v2 = dot(vel,vel);
            if (r2 <= m_R2 || v2 == Scalar(0))
               {
               dt = Scalar(0);
               return false;
               }

            /*
             * Find the time(dt) particle spent outside after collision with the sphere . This time is
             * found by backtracking the position, r* = r-dt*v, and solving for dt when dot(r*,r*) = R0^2.
	     * R0 here in the formula is the radius of container when collision occured(R0 = R - V*dt), 
	     * where R is the radius of sphere at time when it already travelled outside(position r*)
             * This gives a quadratic equation in dt; the smaller root is the solution.
             */

            const Scalar rv = dot(pos,vel);
	    const Scalar RV = m_R*m_V;   //RV is R*V
	    const Scalar rv_RV = rv - RV;  //rv_RV is rv - RV
            dt = (rv_RV – fast::sqrt(rv_RV*rv_RV-(v2-m_V2)*(r2-m_R2)))/(v2-m_V2);

            // backtrack the particle for time dt to get to point of contact
            pos -= vel*dt;

            // update velocity according to boundary conditions
            /*
             * Let n = r/R be the normal unit vector at the point of contact r (the particle position, which has been
             * backtracked to the surface in the previous step).
             * The perpendicular and parallel components of the velocity are:
             * v_perp = (v.n)n = (v.r/R^2)r
             * v_para = v-v_perp
             */
            if (m_bc == boundary::no_slip)
                {
                /* No-slip and no penetration requires reflection of both parallel and perpendicular components 
		 * v'= -vel + V_interface 
                 * This results in just flipping of all the velocity components.
		 * V_v is the velocity of spherical geometry in vector form which is V*normal vector along surface
		 * V_V = V*(normal vector calculated from pos) = V*(pos/sqrt(pos.pos))
                 */
		const Scalar3 V_v = (m_V*pos/(fast::sqrt(r2)));
                vel = -vel + V_v;
                }
            else if (m_bc == boundary::slip)
                {
                /*
                 * Only no-penetration condition is enforced, so only v_perp is reflected.
                 * The new velocity v' is:
                 * v' = -v_perp + v_para + V_interface = v - 2*v_perp + V_interface
                */
		const Scalar3 V_v = (m_V*pos/(fast::sqrt(r2)));
                const Scalar3 vperp = (dot(vel,pos)/m_R2)*pos;
                vel -= Scalar(2)*vperp - V_v;
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
            return dot(pos,pos) > m_R2;
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
            Scalar3 hi;
            Scalar3 lo;
            hi.x = box.getHi().x; hi.y = box.getHi().y; hi.z = box.getHi().z;
            lo.x = box.getLo().x; lo.y = box.getLo().y; lo.z = box.getLo().z;

            return ((hi.x-m_R0) >= cell_size && (-lo.x-m_R0) >= cell_size &&
                    (hi.y-m_R0) >= cell_size && (-lo.y-m_R0) >= cell_size &&
                    (hi.y-m_R0) >= cell_size && (-lo.y-m_R0) >= cell_size );
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
            return std::string("SphereRetracting");
            }
        #endif // NVCC

    private:
        const Scalar m_R;       //!< Sphere radius
        const Scalar m_R2;      //!< Square of sphere radius
        const boundary m_bc;    //!< Boundary condition
	const Scalar m_V;       //!<Velocity of container
	const Scalar m_V2;      //!<square of velocity of spherical container
	const Scalar m_R0;      //!initial radius of spherical container
    };

} // end namespace detail
} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_SPHERE_GEOMETRY_H_
