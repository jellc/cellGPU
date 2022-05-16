// https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.078001
// Supplemental Material

#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexQuadraticEnergy.h"
#include "selfPropelledCellVertexDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
/*!
This file compiles to produce an executable that can be used to reproduce the timing information
for the 2D AVM model found in the "cellGPU" paper, using the following parameters:
i = 1000
t = 4000
e = 0.01
dr = 1.0,
along with a range of v0 and p0. This program also demonstrates the use of brownian dynamics
applied to the vertices themselves.
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
vertex model's computeForces() funciton right before saving a state.
*/

// #define _Brownian
#define S 100

int main(int argc, char*argv[])
{
// Seeds
vector<int> seeds{};
for(int i=0; i<S; ++i){
    seeds.emplace_back(i);
}   
   
   char model[256];
#ifdef _Brownian
    sprintf(model, "brownian");
#else
    sprintf(model, "self-propelled");
#endif

    // ios::sync_with_stdio(0);
    // cin.tie(0);
    // cerr.tie(0);
    // cout.tie(0);


    int numpts = 500; //number of cells
    int Nvert = 2*numpts;
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 100000; //number of time steps to run after initialization
    int initSteps = 2000; //number of initialization steps

    Dscalar zeta = 1;

    Dscalar dt = 0.01; //the time step size
    Dscalar p0 = 4.0;  //the preferred perimeter
    Dscalar a0 = 1;  // the preferred area
    Dscalar v0 = 0.05 / zeta;  // the self-propulsion
    Dscalar Dr = 0.1;  //the rotational diffusion constant of the cell directors
    Dscalar lT1 = 0.01; // threshold for T1 transition
    Dscalar KA = 3 /zeta;
    Dscalar KP = 1 / zeta;
    int program_switch = 0; //various settings control output

    int c;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
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

   char path[256];
   sprintf(path, "data/%s", model);
   char seedf[256];
   sprintf(seedf, "%s/seeds.txt", path);
   fstream seedfile(seedf);
   if(!seedfile){
       cerr<<"Failed to open seedfile\n";
       abort();
    }

   
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


    bool runSPV = true;//setting this to true will relax the random cell positions to something more uniform before running vertex model dynamics


// Simulation for each seed.
   for (int seed : seeds) {
       seedfile << seed << endl;


    //define a vertex model configuration with a quadratic energy functional
    shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,a0,p0,reproducible,runSPV);

// このseedを変えても結果おなじ
// auto vq_seed = 10;
// avm->setSeed(vq_seed);

    //possibly save output in netCDF format
    char dataname[256];
    sprintf(dataname,"%s/n=%d_t=%d_p=%.1f_a=%.1f_KA=%.1f_KP=%.1f_seed=%d.nc", path, numpts,tSteps,p0,a0,KA,KP,seed);
    AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
    
    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    avm->setCellPreferencesUniform(a0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    avm->setv0Dr(v0,Dr);
    //when an edge gets less than this long, perform a simple T1 transition
    avm->setT1Threshold(lT1);

    avm->setModuliUniform(KA,KP);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);

    
    //We will define two potential equations of motion, and choose which later on.

#ifdef _Brownian
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
    bd->setT(v0);
    bd->setSeed(seed);
    sim->addUpdater(bd,avm);
    cout<<"\nBrownian\n\n";
#else
    EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
    spp->setSeed(seed);
    sim->addUpdater(spp, avm);
    cout<<"\nSelf-Propelled\n";
#endif
            cout<<"Noise seed: "<<seed<<"\n\n";


    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
//    sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    avm->reportMeanVertexForce();
    avm->setCellPreferencesUniform(a0,p0);

    // { // Dump (A0, P0)
    // ArrayHandle<Dscalar2> vcn(avm->returnAreaPeriPreferences());
    // cout<<vcn.data[0].x<<" "<<vcn.data[0].y<<endl;
    // };

    ncdat.WriteState(avm); // Before initialization
    cout <<fixed<<setprecision(15)<< "Mean q = " << avm->reportq() << endl;
    //perform some initial time steps. If program_switch < 0, save periodically to a netCDF database
    for (int timestep = 0; timestep < initSteps+1; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            ncdat.WriteState(avm);
            };
        };
    avm->reportMeanVertexForce();
    ncdat.WriteState(avm); // After initialization
    cout <<fixed<<setprecision(15)<< "Mean q = " << avm->reportq() << endl;
    
    // { // Dump (A0, P0)
    // ArrayHandle<Dscalar2> vcn(avm->returnAreaPeriPreferences());
    // cout<<vcn.data[0].x<<" "<<vcn.data[0].y<<endl;
    // };

    //run for additional timesteps, and record timing information. Save frames to a database if desired
    cudaProfilerStart();
    t1=clock();
    for (int timestep = 0; timestep < tSteps; ++timestep)
        {
           // ncdat.WriteState(avm);
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            ncdat.WriteState(avm);
            };
        };
    cudaProfilerStop();

    // { // Dump (A0, P0)
    // ArrayHandle<Dscalar2> vcn(avm->returnAreaPeriPreferences());
    // cout<<vcn.data[0].x<<" "<<vcn.data[0].y<<endl;
    // };

    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;
    avm->reportMeanVertexForce();
    ncdat.WriteState(avm); // Final state
    cout <<fixed<<setprecision(15)<< "Mean q = " << avm->reportq() << endl;

    cout<< "output: "<<dataname<<endl;
   }
    
//     char fs[256],f[256];
//     sprintf(fs, "a%.1f_fs", a0);
//     sprintf(f, "a%.1f_f", a0);

// cout<<avm->forcesUpToDate<<"\n";
//     avm->computeForces();

    //  {
    // freopen(fs, "w", stderr);
    // ArrayHandle<Dscalar2> h_fs(avm->vertexForceSets,access_location::host, access_mode::overwrite);
    // for (int ii = 0; ii < Nvert*3; ++ii)
    //     cerr<<h_fs.data[ii].x<<" "<<h_fs.data[ii].y<<"\n";
    
    // freopen(f, "w", stderr);
    // ArrayHandle<Dscalar2> h_f(avm->vertexForces,access_location::host, access_mode::overwrite);
    // for (int ii = 0; ii < Nvert; ++ii)
    //     cerr<<h_f.data[ii].x<<" "<<h_f.data[ii].y<<"\n";

    // for (int ii = 0; ii < Nvert; ++ii)
    //     {
    //         Dscalar tmpx = 0.0;
    //         Dscalar tmpy = 0.0;
    //         for(int j = ii*3; j<(ii+1)*3; ++j) {
    //             tmpx += h_fs.data[j].x;
    //             tmpy += h_fs.data[j].y;
    //         }
    //         assert(abs(tmpx - h_f.data[ii].x) < 1e-9);
    //         assert(abs(tmpy - h_f.data[ii].y) < 1e-9);
    //     }
    // };
    
    if(initializeGPU)
        cudaDeviceReset();

    cout<<"\nDone!\n";
    return 0;
}

/**
@note
KA ~ 1.0 のときは
A0 の影響がほとんどでない (近傍和で相殺する)
**/