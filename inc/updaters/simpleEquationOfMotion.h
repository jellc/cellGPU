#ifndef simpleEquationOfMotion_H
#define simpleEquationOfMotion_H

#include "curand.h"
#include "curand_kernel.h"
#include "simpleEquationOfMotion.cuh"
#include "gpuarray.h"
#include "updater.h"
#include "noiseSource.h"

/*! \file simpleEquationOfMotion.h */
//!A base class for implementing simple equations of motion
/*!
In cellGPU a "simple" equation of motion is one that can take a GPUArray of forces and return a set
of displacements. A derived class of this might be the self-propelled particle equations of motion,
or simple Brownian dynamics.
Derived classes must implement the integrateEquationsOfMotion function. Additionally, equations of
motion act on a cell configuration, and in general require that the configuration, C,  passed in to the
equation of motion provides access to the following:
C->getNumberOfDegreesOfFreedom() should return the number of degrees of freedom (up to a factor of
dimension)
C->computeForces() should calculate the negative gradient of the energy in whatever model T implements
C->getForces(f) is able to be called after T.computeForces(), and copies the forces to the variable f
C->moveDegreesOfFreedom(disp) moves the degrees of freedom according to the GPUArray of displacements
C->enforceTopology() takes care of any business the model that T implements needs after the
positions of the underlying degrees of freedom have been updated

*/
class simpleEquationOfMotion : public updater
    {
    public:
        //!base constructor sets default time step size
        simpleEquationOfMotion()
            {
            Period = 1;
            Phase = 0;
            deltaT = 0.01; GPUcompute =true;Timestep = 0;
            };
        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion(){};

        //!get the number of timesteps run
        int getTimestep(){return Timestep;};
        //!get the current simulation time
        Dscalar getTime(){return (Dscalar)Timestep * deltaT;};
        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};
        //!Get the number of degrees of freedom of the equation of motion
        int getNdof(){return Ndof;};
        //!Set the number of degrees of freedom of the equation of motion
        void setNdof(int _n){Ndof = _n;};

        //!Set whether the integration of the equations of motion should always use the same random numbers
        void setReproducible(bool rep)
            {
            noise.setReproducible(rep);
            if (GPUcompute)
                noise.initializeGPURNGs(1337,0);
            };

        //!Set the GPU initialization to true
//        void initializeGPU(bool initGPU){initializeGPURNG = initGPU;};
        //! performUpdate just maps to integrateEquationsOfMotion
        virtual void performUpdate(){integrateEquationsOfMotion();};

    protected:
        //! A source of noise for the equation of motion
        noiseSource noise;
        //!The number of degrees of freedom the equations of motion need to know about
        int Ndof;
        //! Count the number of integration timesteps
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;
        //!a vector of the re-indexing information
        vector<int> reIndexing;

        //!an internal GPUArray for holding displacements
        GPUArray<Dscalar2> displacements;

        //!re-index the any RNGs associated with the e.o.m.
        void reIndexRNG(GPUArray<curandState> &array)
            {
            GPUArray<curandState> TEMP = array;
            ArrayHandle<curandState> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<curandState> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < reIndexing.size(); ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
        //!Re-index cell arrays after a spatial sorting has occured.
        void reIndexArray(GPUArray<int> &array)
            {
            GPUArray<int> TEMP = array;
            ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar> &array)
            {
            GPUArray<Dscalar> TEMP = array;
            ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar2> &array)
            {
            GPUArray<Dscalar2> TEMP = array;
            ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
    };

typedef shared_ptr<simpleEquationOfMotion> EOMPtr;
typedef weak_ptr<simpleEquationOfMotion> WeakEOMPtr;

#endif