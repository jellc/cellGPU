#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "analysisPackage.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 400; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 10000; //number of time steps to run after initialization
    int initSteps = 2000; //number of initialization steps

    Dscalar zeta = 1;

    Dscalar dt = 0.01; //the time step size
    Dscalar p0 = 4.0;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 0.05 / zeta;  // the self-propulsion
    Dscalar Dr = 1.0;  //the rotational diffusion constant of the cell directors
    Dscalar lT1 = 0.01; // threshold for T1 transition
    Dscalar KA = 1.0 /zeta;
    Dscalar KP = 1.0 / zeta;

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    clock_t t1,t2; //clocks for timing information
    bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
    else
        initializeGPU = false;

    //define an equation of motion object...here for self-propelled cells
    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> spv  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    spv->setv0Dr(v0,Dr);


    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp,spv);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    printf("Finished with initialization\n");
    cout << "current q = " << spv->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    spv->reportMeanCellForce(false);
    if(!initializeGPU)
        spv->setCPU(false);//turn off globabl-ony mode

    //run for additional timesteps, compute dynamical features, and record timing information
    dynamicalFeatures dynFeat(spv->returnPositions(),spv->Box);
    logSpacedIntegers logInts(0,0.05);
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        //if(ii%100 ==0)
        if(ii == logInts.nextSave)
            {
            printf("timestep %i\t\t energy %f \t msd %f \t overlap %f topoUpdates %i \n",ii,spv->computeEnergy(),dynFeat.computeMSD(spv->returnPositions()),dynFeat.computeOverlapFunction(spv->returnPositions()),spv->localTopologyUpdates);
            logInts.update();
            };
        sim->performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << spv->reportq() << endl;
    cout << "number of local topology updates per cell per tau = " << spv->localTopologyUpdates*(1.0/numpts)*(1.0/tSteps/dt) << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
