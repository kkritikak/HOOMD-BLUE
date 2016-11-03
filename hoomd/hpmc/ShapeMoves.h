#ifndef _SHAPE_MOVES_H
#define _SHAPE_MOVES_H

#include <hoomd/extern/saruprng.h>
#include "ShapeUtils.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/Eigen/Dense>

namespace hpmc {

template < typename Shape, typename RNG>
class shape_move_function
{
public:
    shape_move_function(unsigned int ntypes) : m_determinantInertiaTensor(0), m_step_size(ntypes) {}
    shape_move_function(const shape_move_function& src) : m_determinantInertiaTensor(src.getDeterminantInertiaTensor()), m_step_size(src.getStepSize()) {}

    //! prepare is called at the beginning of every update()
    virtual void prepare(unsigned int timestep) = 0;

    //! construct is called for each particle type that will be changed in update()
    virtual void construct(const unsigned int&, const unsigned int&, typename Shape::param_type&, RNG&) = 0;

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(const unsigned int) = 0;

    virtual Scalar getParam(size_t i ) { return 0.0; }

    virtual size_t getNumParam() { return 0; }

    Scalar getDeterminant() const { return m_determinantInertiaTensor; }

    Scalar getStepSize(const unsigned int& type_id) const { return m_step_size[type_id]; }

    void setStepSize(const unsigned int& type_id, const Scalar& stepsize) { m_step_size[type_id] = stepsize; }

protected:
    Scalar                          m_determinantInertiaTensor;     // TODO: REMOVE?
    std::vector<Scalar>             m_step_size;                    // maximum stepsize. input/output
};

template < typename Shape, typename RNG >
class python_callback_parameter_shape_move : public shape_move_function<Shape, RNG>
{
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
    using shape_move_function<Shape, RNG>::m_step_size;
public:
    python_callback_parameter_shape_move(   unsigned int ntypes,
                                            pybind11::object python_function,
                                            std::vector< std::vector<Scalar> > params,
                                            std::vector<Scalar> stepsize,
                                            Scalar mixratio
                                        )
        :  shape_move_function<Shape, RNG>(ntypes), m_params(params), m_python_callback(python_function) //, m_normalized(normalized)
        {
        if(m_step_size.size() != stepsize.size())
            throw std::runtime_error("must provide a stepsize for each type");

        m_step_size = stepsize;
        m_select_ratio = fmin(mixratio, 1.0)*65535;
        m_determinantInertiaTensor = 0.0;
        }

    void prepare(unsigned int timestep)
        {
        m_params_backup = m_params;
        // m_step_size_backup = m_step_size;
        }

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& shape, RNG& rng)
        {
        // gonna make the move.
        // Saru rng(m_select_ratio, m_seed, timestep);
        // type_id = type_id;
        for(size_t i = 0; i < m_params[type_id].size(); i++)
            {
            Scalar x = ((rng.u32() & 0xffff) < m_select_ratio) ? rng.s(fmax(-m_step_size[type_id], -(m_params[type_id][i])), fmin(m_step_size[type_id], (1.0-m_params[type_id][i]))) : 0.0;
            m_params[type_id][i] += x;
            }

        // if(m_normalized)
        //     {
        //     }
        // else
        //     {
        //     // could gut this part becuase of the other functions.
        //     for(size_t i = 0; i < m_params[type_id].size() && (m_params[type_id].size()%3 == 0); i+=3)
        //         {
        //         if( (rng.u32()& 0xffff) < m_select_ratio )
        //             {
        //             Scalar x = rng.s(-1.0, 1.0);
        //             Scalar y = rng.s(-1.0, 1.0);
        //             Scalar z = rng.s(-1.0, 1.0);
        //             Scalar mag = rng.s(0.0, m_step_size[type_id])/sqrt(x*x + y*y + z*z);
        //             m_params[type_id][i] += x*mag;
        //             m_params[type_id][i+1] += y*mag;
        //             m_params[type_id][i+2] += z*mag;
        //             }
        //         }
        //     }

        pybind11::object shape_data = m_python_callback(m_params[type_id]);
        shape = pybind11::cast< typename Shape::param_type >(shape_data);
        detail::mass_properties<Shape> mp(shape);
        m_determinantInertiaTensor = mp.getDeterminant();
        // m_scale = Scalar(1.0);
        // if(!m_normalized)
        //     {
        //     m_scale = pybind11::cast< Scalar >(shape_data[2]); // only extract if we have to.
        //     detail::shape_param_to_vector<Shape> converter;
        //     converter(shape, m_params[type_id]);
        //     }
        // m_step_size[type_id] *= m_scale; // only need to scale if the parameters are not normalized
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        std::swap(m_params, m_params_backup);
        // std::swap(m_step_size, m_step_size_backup);
        }

    Scalar getParam(size_t k)
        {
        size_t n = 0;
        for (size_t i = 0; i < m_params.size(); i++)
            {
            size_t next = n + m_params[i].size();
            if(k < next)
                return m_params[i][k - n];
            n = next;
            }

        return 0.0; // out of range.
        }

    size_t getNumParam()
        {
        size_t n = 0;
        for (size_t i = 0; i < m_params.size(); i++)
            n += m_params[i].size();
        return n;
        }

private:
    std::vector<Scalar>                     m_step_size_backup;
    unsigned int                            m_select_ratio;     // fraction of parameters to change in each move. internal use
    Scalar                                  m_scale;            // the scale needed to keep the particle at constant volume. internal use
    std::vector< std::vector<Scalar> >      m_params_backup;    // all params are from 0,1
    std::vector< std::vector<Scalar> >      m_params;           // all params are from 0,1
    pybind11::object                        m_python_callback;  // callback that takes m_params as an argiment and returns (shape, det(I))
    // bool                                    m_normalized;       // if true all parameters are restricted to (0,1)
};

template< typename Shape, typename RNG >
class constant_shape_move : public shape_move_function<Shape, RNG>
{
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
public:
    constant_shape_move(const unsigned int& ntypes, const std::vector< typename Shape::param_type >& shape_move) : shape_move_function<Shape, RNG>(ntypes), m_shapeMoves(shape_move)
        {
        if(ntypes != m_shapeMoves.size())
            throw std::runtime_error("Must supply a shape move for each type");
        for(size_t i = 0; i < m_shapeMoves.size(); i++)
            {
            detail::mass_properties<Shape> mp(m_shapeMoves[i]);
            m_determinants.push_back(mp.getDeterminant());
            }
        }

    void prepare(unsigned int timestep) {}

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& shape, RNG& rng)
        {
        shape = m_shapeMoves[type_id];
        m_determinantInertiaTensor = m_determinants[type_id];
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        }

private:
    std::vector< typename Shape::param_type >   m_shapeMoves;
    std::vector< Scalar >                       m_determinants;
};

template < typename ShapeConvexPolyhedronType, typename RNG >
class convex_polyhedron_generalized_shape_move : public shape_move_function<ShapeConvexPolyhedronType, RNG>
{
    using shape_move_function<ShapeConvexPolyhedronType, RNG>::m_determinantInertiaTensor;
    using shape_move_function<ShapeConvexPolyhedronType, RNG>::m_step_size;
public:
    convex_polyhedron_generalized_shape_move(
                                            unsigned int ntypes,
                                            Scalar stepsize,
                                            Scalar mixratio,
                                            Scalar volume
                                        ) : shape_move_function<ShapeConvexPolyhedronType, RNG>(ntypes), m_volume(volume)
        {
        // if(m_step_size.size() != stepsize.size())
        //     throw std::runtime_error("must provide a stepsize for each type");

        m_determinantInertiaTensor = 1.0;
        m_scale = 1.0;
        m_step_size.clear();
        m_step_size.resize(ntypes, stepsize);
        m_select_ratio = fmin(mixratio, 1.0)*65535;
        }

    void prepare(unsigned int timestep)
        {
        m_step_size_backup = m_step_size;
        }

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename ShapeConvexPolyhedronType::param_type& shape, RNG& rng)
        {
        // type_id = type_id;
        // mix the shape.
        for(size_t i = 0; i < shape.N; i++)
            {
            if( (rng.u32()& 0xffff) < m_select_ratio )
                {
                Scalar x = rng.s(-1.0, 1.0);
                Scalar y = rng.s(-1.0, 1.0);
                Scalar z = rng.s(-1.0, 1.0);
                Scalar mag = rng.s(0.0, m_step_size[type_id])/sqrt(x*x + y*y + z*z);
                shape.x[i] += x*mag;
                shape.y[i] += y*mag;
                shape.z[i] += z*mag;
                }
            }

        detail::ConvexHull convex_hull(shape); // compute the convex_hull.
        convex_hull.compute();
        detail::mass_properties<ShapeConvexPolyhedronType> mp(convex_hull.getPoints(), convex_hull.getFaces());
        Scalar volume = mp.getVolume();
        vec3<Scalar> centroid = mp.getCenterOfMass();
        m_scale = pow(m_volume/volume, 1.0/3.0);
        Scalar rsq = 0.0;
        std::vector< vec3<Scalar> > points(shape.N);
        for(size_t i = 0; i < shape.N; i++)
            {
            shape.x[i] -= centroid.x;
            shape.x[i] *= m_scale;
            shape.y[i] -= centroid.y;
            shape.y[i] *= m_scale;
            shape.z[i] -= centroid.z;
            shape.z[i] *= m_scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            points[i] = vert;
            }
        detail::mass_properties<ShapeConvexPolyhedronType> mp2(points, convex_hull.getFaces());
        m_determinantInertiaTensor = mp2.getDeterminant();
        shape.diameter = 2.0*sqrt(rsq);
        m_step_size[type_id] *= m_scale; // only need to scale if the parameters are not normalized
        }

    // void advance(unsigned int timestep)
    //     {
    //     // nothing to do.
    //     }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        std::swap(m_step_size, m_step_size_backup);
        }

private:
    std::vector<Scalar>     m_step_size_backup;
    unsigned int            m_select_ratio;
    Scalar                  m_scale;
    Scalar                  m_volume;
};

template <class Shape, class RNG>
struct shear
    {
    shear(Scalar) {}
    void operator() (typename Shape::param_type& param, RNG& rng)
        {
        throw std::runtime_error("shear is not implemented for this shape.");
        }
    };

template <class Shape, class RNG>
struct scale
    {
    bool isotropic;
    scale(bool iso = true) : isotropic(iso) {}
    void operator() (typename Shape::param_type& param, RNG& rng)
        {
        throw std::runtime_error("scale is not implemented for this shape.");
        }
    };


template <unsigned int max_verts, class RNG>
struct shear< ShapeConvexPolyhedron<max_verts>, RNG >
    {
    Scalar shear_max;
    shear(Scalar smax) : shear_max(smax) {}
    void operator() (typename ShapeConvexPolyhedron<max_verts>::param_type& param, RNG& rng)
        {
        Scalar gamma = rng.s(-shear_max, shear_max), gammaxy = 0.0, gammaxz = 0.0, gammayz = 0.0, gammayx = 0.0, gammazx = 0.0, gammazy = 0.0;
        int dim = int(6*rng.s(0.0, 1.0));
        if(dim == 0) gammaxy = gamma;
        else if(dim == 1) gammaxz = gamma;
        else if(dim == 2) gammayz = gamma;
        else if(dim == 3) gammayx = gamma;
        else if(dim == 4) gammazx = gamma;
        else if(dim == 5) gammazy = gamma;
        Scalar dsq = 0.0;
        for(unsigned int i = 0; i < param.N; i++)
            {
            param.x[i] = param.x[i] + param.y[i]*gammaxy + param.z[i]*gammaxz;
            param.y[i] = param.x[i]*gammayx + param.y[i] + param.z[i]*gammayz;
            param.z[i] = param.x[i]*gammazx + param.y[i]*gammazy + param.z[i];
            vec3<Scalar> vert( param.x[i], param.y[i], param.z[i]);
            dsq = fmax(dsq, dot(vert, vert));
            }
        param.diameter = 2.0*sqrt(dsq);
        // std::cout << "shearing by " << gamma << std::endl;
        }
    };

template <unsigned int max_verts, class RNG>
struct scale< ShapeConvexPolyhedron<max_verts>, RNG >
    {
    bool isotropic;
    Scalar scale_min;
    Scalar scale_max;
    scale(Scalar movesize, bool iso = true) : isotropic(iso)
        {
        if(movesize < 0.0 || movesize > 1.0)
            {
            movesize = 0.0;
            }
        scale_max = (1.0+movesize);
        scale_min = 1.0/scale_max;
        }
                 // () name of perator and second (...) are the parameters
                 //  You can overload the () operator to call your object as if it was a function
    void operator() (typename ShapeConvexPolyhedron<max_verts>::param_type& param, RNG& rng)
        {
        Scalar sx, sy, sz;
        Scalar s = rng.s(scale_min, scale_max);
        sx = sy = sz = s;
        if(!isotropic)
            {
            sx = sy = sz = 1.0;
            Scalar dim = rng.s(0.0, 1.0);
            if (dim < 1.0/3.0) sx = s;
            else if (dim < 2.0/3.0) sy = s;
            else sz = s;
            }
        for(unsigned int i = 0; i < param.N; i++)
            {
            param.x[i] *= sx;
            param.y[i] *= sy;
            param.z[i] *= sz;
            }
        param.diameter *= s;
        // std::cout << "scaling by " << s << std::endl;
        }
    };

template < class RNG >
class scale< ShapeEllipsoid, RNG >
{
    const Scalar m_v;
    const Scalar m_v1;
    const Scalar m_min;
    const Scalar m_max;
public:
    scale(Scalar movesize, bool) : m_v(1.0), m_v1(M_PI*4.0/3.0), m_min(-movesize), m_max(movesize) {}
    void operator ()(ShapeEllipsoid::param_type& param, RNG& rng)
        {
        Scalar lnx = log(param.x/param.y);
        Scalar dx = rng.s(m_min, m_max);
        Scalar x = fast::exp(lnx+dx);
        Scalar b = pow(m_v/m_v1/x, 1.0/3.0);

        param.x = x*b;
        param.y = b;
        param.z = b;
        }
};



template<class Shape, class RNG>
class elastic_shape_move_function : public shape_move_function<Shape, RNG>
{  // Derived class from shape_move_function base class
    // using shape_move_function<Shape, RNG>::m_shape;
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
    using shape_move_function<Shape, RNG>::m_step_size;
    // using shape_move_function<Shape, RNG>::m_scale;
    // using shape_move_function<Shape, RNG>::m_select_ratio;
    std::vector <Eigen::Matrix3f> m_eps;
    std::vector <Eigen::Matrix3f> m_Fbar;
    //Scalar a_max= 0.01;
public:
    elastic_shape_move_function(
                                    unsigned int ntypes,
                                    const Scalar& stepsize,
                                    Scalar move_ratio
                                ) : shape_move_function<Shape, RNG>(ntypes), m_mass_props(ntypes)
        {
        m_select_ratio = fmin(move_ratio, 1.0)*65535;
        m_step_size.resize(ntypes, stepsize);
        m_eps.resize(ntypes, Eigen::Matrix3f::Identity());
        m_Fbar.resize(ntypes, Eigen::Matrix3f::Identity());
        std::fill(m_step_size.begin(), m_step_size.end(), stepsize);
        //m_Fbar = Eigen::Matrix3f::Identity();
        m_determinantInertiaTensor = 1.0;
        }

    void prepare(unsigned int timestep) { /* Nothing to do. */ }
    //! construct is called at the beginning of every update()                                            # param was shape - Luis
    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& param, RNG& rng)
        {
        using Eigen::Matrix3f;
        // unsigned int move_type_select = rng.u32() & 0xffff;
        // Saru rng(m_select_ratio, m_seed, timestep);
        /*if( move_type_select < m_select_ratio)
            {
            scale<Shape, RNG> move(m_step_size[type_id], false);
            move(shape, rng); // always make the move
            }
        else
            {
            shear<Shape, RNG> move(m_step_size[type_id]);
            move(shape, rng); // always make the move
            }
        m_mass_props[type_id].updateParam(shape, false); // update allows caching since for some shapes a full compute is not necessary.
        m_determinantInertiaTensor = m_mass_props[type_id].getDeterminant(); */
        
        // look for Saru
        //
          Matrix3f I(3,3);
          Matrix3f alpha(3,3);
          Matrix3f F(3,3), Fbar(3,3);
          Matrix3f eps(3,3), E(3,3);
          
          // TODO: Define I as global?        
          I << 1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0;  
          
          for(int i=0;i<3;i++)
          { 
            for (int j=0;j<3;j++)
            {
              alpha(i,j) =  rng.s(-m_step_size[type_id], m_step_size[type_id]);
              std::cout << alpha(i,j) << std::endl;
            }
          }
         //std::cout << "alpha_max = " << a_max << std::endl;
         F = I + alpha;
         std::cout << "det(F) = " << F.determinant() << std::endl;
         Fbar = F / pow(F.determinant(),1.0/3.0);
         std::cout << "det(Fbar) = " << Fbar.determinant() << std::endl;
         m_Fbar[type_id] = Fbar*m_Fbar[type_id];
         eps = 0.5*(m_Fbar[type_id].transpose() + m_Fbar[type_id]) - I ; 
         //E   = 0.5(Fbar.transpose() * Fbar - I) ; for future reference
        std::cout << Fbar << std::endl;
        for(unsigned int i = 0; i < param.N; i++)
            {
            param.x[i] = Fbar(1,1)*param.x[i] + Fbar(1,2)*param.y[i] + Fbar(1,3)*param.z[i];
            param.y[i] = Fbar(2,1)*param.x[i] + Fbar(2,2)*param.y[i] + Fbar(2,3)*param.z[i];
            param.z[i] = Fbar(3,1)*param.x[i] + Fbar(3,2)*param.y[i] + Fbar(3,3)*param.z[i];

            vec3<Scalar> vert( param.x[i], param.y[i], param.z[i]);
            //dsq = fmax(dsq, dot(vert, vert));
            }
        m_eps[type_id] = eps;
        }
        Eigen::Matrix3f getEps(unsigned int type_id){
           return m_eps[type_id];
         }
    //! advance whenever the proposed move is accepted.
    // void advance(unsigned int timestep){ /* Nothing to do. */ }

    //! retreat whenever the proposed move is rejected.
    void retreat(unsigned int timestep){ /* Nothing to do. */ }

protected:
    unsigned int            m_select_ratio;
    std::vector< detail::mass_properties<Shape> > m_mass_props;
};

template<class Shape>
class ShapeLogBoltzmannFunction
{
public:
    virtual Scalar operator()(const unsigned int& N, const unsigned int type_id, const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold) { throw std::runtime_error("not implemented"); return 0.0;}
};

template<class Shape>
class AlchemyLogBoltzmannFunction : public ShapeLogBoltzmannFunction<Shape>
{
public:
    virtual Scalar operator()(const unsigned int& N,const unsigned int type_id, const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold)
        {
        return (Scalar(N)/Scalar(2.0))*log(inew/iold);
        }
};

template< class Shape >
class ShapeSpringBase : public ShapeLogBoltzmannFunction<Shape>
{
protected:
    Scalar m_k;
    Scalar m_volume;
    std::unique_ptr<typename Shape::param_type> m_reference_shape;
public:
    ShapeSpringBase(Scalar k, typename Shape::param_type shape) : m_k(k), m_reference_shape(new typename Shape::param_type)
    {
        (*m_reference_shape) = shape;
        detail::mass_properties<Shape> mp(*m_reference_shape);
        m_volume = mp.getVolume();
    }
};

/*template <typename Shape> class ShapeSpring : public ShapeSpringBase<Shape> { Empty base template will fail on export to python. };

template <>
class ShapeSpring<ShapeEllipsoid> : public ShapeSpringBase<ShapeEllipsoid>
{
    using ShapeSpringBase<ShapeEllipsoid>::m_k;
    using ShapeSpringBase<ShapeEllipsoid>::m_reference_shape;
public:
    ShapeSpring(Scalar k, ShapeEllipsoid::param_type ref) : ShapeSpringBase<ShapeEllipsoid>(k, ref) {}
    Scalar operator()(const unsigned int& N, const ShapeEllipsoid::param_type& shape_new, const Scalar& inew, const ShapeEllipsoid::param_type& shape_old, const Scalar& iold)
        {
        //TODO: this uses the sphere as the reference. modify to use the reference shape.
        Scalar x_new = shape_new.x/shape_new.y;
        Scalar x_old = shape_old.x/shape_old.y;
        return m_k*(log(x_old)*log(x_old) - log(x_new)*log(x_new)); // -\beta dH
        }
};*/

template<class Shape>
class ShapeSpring : public ShapeSpringBase< Shape >
{
    using ShapeSpringBase< Shape >::m_k;
    
    using ShapeSpringBase< Shape >::m_reference_shape;
    using ShapeSpringBase< Shape >::m_volume;
    //using elastic_shape_move_function<Shape, Saru>;
    std::shared_ptr<elastic_shape_move_function<Shape, Saru> > m_shape_move;
public:
    ShapeSpring(Scalar k, typename Shape::param_type ref, std::shared_ptr<elastic_shape_move_function<Shape, Saru> > P) : ShapeSpringBase <Shape> (k, ref ) , m_shape_move(P)
        {
        }
    Scalar operator()(const unsigned int& N, const unsigned int type_id ,const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold)
        {
          //using Eigen::Matrix3f;
          Eigen::Matrix3f eps = m_shape_move->getEps(type_id);
          AlchemyLogBoltzmannFunction< Shape > fn;
          //Scalar dv;
          Scalar e_ddot_e = 0.0;
          detail::mass_properties<Shape> mp(shape_new);
          //dv = mp.getVolume()-m_volume;
          e_ddot_e = eps(1,1)*eps(1,1) + eps(1,2)*eps(2,1) + eps(1,3)*eps(3,1) + 
                     eps(2,1)*eps(1,2) + eps(2,2)*eps(2,2) + eps(2,3)*eps(3,2) + 
                     eps(3,1)*eps(1,3) + eps(3,2)*eps(2,3) + eps(3,3)*eps(3,3) ;

        /*  for (unsigned int i=0; i<3;i++)
          { 
           for (unsigned int j=0; j<3;i++)
           {
              eps_ddot += eps(i,j)*eps(j,i) 
           }
          } */
          //std::cout << "Particle volume = " << m_volume << std::endl ; OK
          std::cout << "Stiffness = " << m_k << std::endl ;
          std::cout << "eps ddot eps = " << e_ddot_e << std::endl ;  
          return m_k*e_ddot_e*m_volume + fn(N,type_id,shape_new, inew, shape_old, iold); // -\beta dH
        }
};

//** Python export functions and additional classes to wrap the move and boltzmann interface.
//**
//**
//**
//**
// ! Wrapper class for wrapping pure virtual methods
template<class Shape, class RNG>
class shape_move_function_wrap : public shape_move_function<Shape, RNG>
    {
    public:
        //! Constructor
        shape_move_function_wrap(unsigned int ntypes) : shape_move_function<Shape, RNG>(ntypes) {}
        void prepare(unsigned int timestep)
            {
            PYBIND11_OVERLOAD_PURE( void,                                       /* Return type */
                                    shape_move_function<Shape, RNG>,            /* Parent class */
                                    &shape_move_function<Shape, RNG>::prepare,  /* Name of function */
                                    timestep);                                  /* Argument(s) */
            }

        void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& shape, RNG& rng)
            {
            PYBIND11_OVERLOAD_PURE( void,                                       /* Return type */
                                    shape_move_function<Shape, RNG>,            /* Parent class */
                                    &shape_move_function<Shape, RNG>::construct,/* Name of function */
                                    timestep,                                   /* Argument(s) */
                                    type_id,
                                    shape,
                                    rng);
            }

        void retreat(unsigned int timestep)
            {
            PYBIND11_OVERLOAD_PURE( void,                                       /* Return type */
                                    shape_move_function<Shape, RNG>,            /* Parent class */
                                    &shape_move_function<Shape, RNG>::retreat,  /* Name of function */
                                    timestep);                                  /* Argument(s) */
            }
    };

template<class Shape>
void export_ShapeMoveInterface(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ScaleShearShapeMove(pybind11::module& m, const std::string& name);

template< typename Shape >
void export_ShapeLogBoltzmann(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ShapeSpringLogBoltzmannFunction(pybind11::module& m, const std::string& name);

template<class Shape>
void export_AlchemyLogBoltzmannFunction(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ConvexPolyhedronGeneralizedShapeMove(pybind11::module& m, const std::string& name);

template<class Shape>
void export_PythonShapeMove(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ConstantShapeMove(pybind11::module& m, const std::string& name);

}

#endif
