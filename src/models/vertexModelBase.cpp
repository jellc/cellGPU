#define ENABLE_CUDA

#include "vertexModelBase.h"
#include "vertexModelBase.cuh"
#include "utilities.cuh"
/*! \file vertexModelBase.cpp */

/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void vertexModelBase::initializeVertexModelBase(int n)
    {
    //call initializer chain...sets Ncells = n
    initializeSimpleVertexModelBase(n);

    initializeEdgeFlipLists();

    growCellVertexListAssist.resize(1);
    ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::overwrite);
    h_grow.data[0]=0;
    };

/*!
 *When sortPeriod < 0 this routine does not get called
 \post vertices are re-ordered according to a Hilbert sorting scheme, cells are reordered according
 to what vertices they are near, and all data structures are updated
 */
void vertexModelBase::spatialSorting()
    {
    //call the simpleVertexModelBase to get the basic vertex sorting done
    simpleVertexModelBase::spatialSorting();

    //and then take care of cell and neighbor structures
    GPUArray<int> TEMP_vertexNeighbors = vertexNeighbors;
    GPUArray<int> TEMP_vertexCellNeighbors = vertexCellNeighbors;
    GPUArray<int> TEMP_cellVertices = cellVertices;
    ArrayHandle<int> temp_vn(TEMP_vertexNeighbors,access_location::host, access_mode::read);
    ArrayHandle<int> temp_vcn(TEMP_vertexCellNeighbors,access_location::host, access_mode::read);
    ArrayHandle<int> temp_cv(TEMP_cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> vn(vertexNeighbors,access_location::host, access_mode::readwrite);
    ArrayHandle<int> vcn(vertexCellNeighbors,access_location::host, access_mode::readwrite);
    ArrayHandle<int> cv(cellVertices,access_location::host, access_mode::readwrite);
    ArrayHandle<int> cvn(cellVertexNum,access_location::host,access_mode::read);

    //Great, now use the vertex ordering to derive a cell spatial ordering
    vector<pair<int,int> > idxCellSorter(Ncells);

    vector<bool> cellOrdered(Ncells,false);
    int cellOrdering = 0;
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        if(cellOrdering == Ncells) continue;
        int vertexIndex = ittVertex[vv];
        for (int ii = 0; ii < 3; ++ii)
            {
            int cellIndex = vcn.data[3*vertexIndex +ii];
            if(!cellOrdered[cellIndex])
                {
                cellOrdered[cellIndex] = true;
                idxCellSorter[cellIndex].first=cellOrdering;
                idxCellSorter[cellIndex].second = cellIndex;
                cellOrdering += 1;
                };
            };
        };
    sort(idxCellSorter.begin(),idxCellSorter.end());
    //update tti and itt
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int newidx = idxCellSorter[ii].second;
        itt[ii] = newidx;
        tti[newidx] = ii;
        };
    //update points, idxToTag, and tagToIdx
    vector<int> tempiCell = idxToTag;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxToTag[ii] = tempiCell[itt[ii]];
        tagToIdx[tempiCell[itt[ii]]] = ii;
        };
    //update cell property structures
    cellStructureSort();
    spatiallySortCellActivity();
    reIndexCellArray(cellVertexNum);

    //Finally, now that both cell and vertex re-indexing is known, update auxiliary data structures
    //Start with everything that can be done with just the cell indexing
    //Now the rest
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int vertexIndex = ttiVertex[vv];
        for (int ii = 0; ii < 3; ++ii)
            {
            vn.data[3*vertexIndex+ii] = ttiVertex[temp_vn.data[3*vv+ii]];
            vcn.data[3*vertexIndex+ii] = tti[temp_vcn.data[3*vv+ii]];
            };
        };

    for (int cc = 0; cc < Ncells; ++cc)
        {
        int cellIndex = tti[cc];
        //the cellVertexNeigh array is already sorted
        int neighs = cvn.data[cellIndex];
        for (int nn = 0; nn < neighs; ++nn)
            cv.data[cellNeighborIndexer(nn,cellIndex)] = ttiVertex[temp_cv.data[cellNeighborIndexer(nn,cc)]];
        };
    };

/*!
enforce and update topology of vertex wiring on either the GPU or CPU
*/
void vertexModelBase::enforceTopology()
    {
    if(GPUcompute)
        {
        //see if vertex motion leads to T1 transitions...ONLY allow one transition per vertex and
        //per cell per timestep
        testAndPerformT1TransitionsGPU();
        }
    else
        {
        //see if vertex motion leads to T1 transitions
        testAndPerformT1TransitionsCPU();
        };
    };

/*!
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures
(the vertices)...compute the area and perimeter of the cells
*/
void vertexModelBase::computeGeometryCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);

    //compute the geometry for each cell
    for (int i = 0; i < Ncells; ++i)
        {
        int neighs = h_nn.data[i];
//      Define the vertices of a cell relative to some (any) of its verties to take care of periodic boundaries
        Dscalar2 cellPos = h_v.data[h_n.data[cellNeighborIndexer(neighs-2,i)]];
        Dscalar2 vlast, vcur,vnext;
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        //compute the vertex position relative to the cell position
        vlast.x=0.;vlast.y=0.0;
        int vidx = h_n.data[cellNeighborIndexer(neighs-1,i)];
        Box->minDist(h_v.data[vidx],cellPos,vcur);
        for (int nn = 0; nn < neighs; ++nn)
            {
            //for easy force calculation, save the current, last, and next vertex position in the approprate spot.
            int forceSetIdx= -1;
            for (int ff = 0; ff < 3; ++ff)
                if(h_vcn.data[3*vidx+ff]==i)
                    forceSetIdx = 3*vidx+ff;
            vidx = h_n.data[cellNeighborIndexer(nn,i)];
            Box->minDist(h_v.data[vidx],cellPos,vnext);

            //contribution to cell's area is
            // 0.5* (vcur.x+vnext.x)*(vnext.y-vcur.y)
            Varea += SignedPolygonAreaPart(vcur,vnext);
            Dscalar dx = vcur.x-vnext.x;
            Dscalar dy = vcur.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            //save vertex positions in a convenient form
            h_vc.data[forceSetIdx] = vcur;
            h_vln.data[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
            //advance the loop
            vlast = vcur;
            vcur = vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

/*!
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures (the vertices)
*/
void vertexModelBase::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,      access_location::device,access_mode::read);
    ArrayHandle<int>      d_cvn(cellVertexNum,       access_location::device,access_mode::read);
    ArrayHandle<int>      d_cv(cellVertices,         access_location::device,access_mode::read);
    ArrayHandle<int>      d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,             access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,       access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,            access_location::device,access_mode::overwrite);

    gpu_vm_geometry(
                    d_v.data,
                    d_cvn.data,
                    d_cv.data,
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    Ncells,cellNeighborIndexer,*(Box));
    };

/*!
 Initialize the auxilliary edge flip data structures to zero
 */
void vertexModelBase::initializeEdgeFlipLists()
    {
    vertexEdgeFlips.resize(3*Nvertices);
    vertexEdgeFlipsCurrent.resize(3*Nvertices);
    ArrayHandle<int> h_vflip(vertexEdgeFlips,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vflipc(vertexEdgeFlipsCurrent,access_location::host,access_mode::overwrite);
    for (int i = 0; i < 3*Nvertices; ++i)
        {
        h_vflip.data[i]=0;
        h_vflipc.data[i]=0;
        }

    finishedFlippingEdges.resize(2);
    ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::overwrite);
    h_ffe.data[0]=0;
    h_ffe.data[1]=0;
    };

/*!
A utility function for the CPU T1 transition routine. Given two vertex indices representing an edge that will undergo
a T1 transition, return in the pass-by-reference variables a helpful representation of the cells in the T1
and the vertices to be re-wired...see the comments in "testAndPerformT1TransitionsCPU" for what that representation is
*/
void vertexModelBase::getCellVertexSetForT1(int vertex1, int vertex2, int4 &cellSet, int4 &vertexSet, bool &growList)
    {
    int cell1,cell2,cell3,ctest;
    int vlast, vcur, vnext, cneigh;
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    cell1 = h_vcn.data[3*vertex1];
    cell2 = h_vcn.data[3*vertex1+1];
    cell3 = h_vcn.data[3*vertex1+2];
    //cell_l doesn't contain vertex 1, so it is the cell neighbor of vertex 2 we haven't found yet
    for (int ff = 0; ff < 3; ++ff)
        {
        ctest = h_vcn.data[3*vertex2+ff];
        if(ctest != cell1 && ctest != cell2 && ctest != cell3)
            cellSet.w=ctest;
        };
    //find vertices "c" and "d"
    cneigh = h_cvn.data[cellSet.w];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cellSet.w) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cell1)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };

    //classify cell1
    cneigh = h_cvn.data[cell1];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cell1) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cell1) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cell1)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell1;
    else if(vnext == vertex2)
        cellSet.z = cell1;
    else
        {
        cellSet.y = cell1;
        };

    //classify cell2
    cneigh = h_cvn.data[cell2];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cell2) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cell2) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cell2)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell2;
    else if(vnext == vertex2)
        cellSet.z = cell2;
    else
        {
        cellSet.y = cell2;
        };

    //classify cell3
    cneigh = h_cvn.data[cell3];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cell3) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cell3) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cell3)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell3;
    else if(vnext == vertex2)
        cellSet.z = cell3;
    else
        {
        cellSet.y = cell3;
        };

    //get the vertexSet by examining cells j and l
    cneigh = h_cvn.data[cellSet.y];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cellSet.y) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cellSet.y) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cellSet.y)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.x=vlast;
    vertexSet.y=vnext;
    cneigh = h_cvn.data[cellSet.w];
    vlast = h_cv.data[ cellNeighborIndexer(cneigh-2,cellSet.w) ];
    vcur = h_cv.data[ cellNeighborIndexer(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[cellNeighborIndexer(cn,cellSet.w)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.w=vlast;
    vertexSet.z=vnext;

    //Does the cell-vertex-neighbor data structure need to be bigger...for safety check all cell-vertex numbers, even if it won't be incremented?
    if(h_cvn.data[cellSet.x] == vertexMax || h_cvn.data[cellSet.y] == vertexMax || h_cvn.data[cellSet.z] == vertexMax || h_cvn.data[cellSet.w] == vertexMax)
        growList = true;
    };

/*!
Test whether a T1 needs to be performed on any edge by simply checking if the edge length is beneath a threshold.
This function also performs the transition and maintains the auxiliary data structures
 */
void vertexModelBase::testAndPerformT1TransitionsCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vn(vertexNeighbors,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_cv(cellVertices,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::readwrite);

    Dscalar2 edge;
    //first, scan through the list for any T1 transitions...
    int vertex2;
    //keep track of whether vertexMax needs to be increased
    int vMax = vertexMax;
    /*
     The following is the convention:
     cell i: contains both vertex 1 and vertex 2, in CW order
     cell j: contains only vertex 1
     cell k: contains both vertex 1 and vertex 2, in CCW order
     cell l: contains only vertex 2
     */
    int4 cellSet;
    /*
    vertexSet (a,b,c,d) have those indices in which before the transition
    cell i has CCW vertices: ..., c, v2, v1, a, ...
    and
    cell k has CCW vertices: ..., b,v1,v2,d, ...
    */
    int4 vertexSet;
    Dscalar2 v1,v2;
    for (int vertex1 = 0; vertex1 < Nvertices; ++vertex1)
        {
        v1 = h_v.data[vertex1];
        //look at vertexNeighbors list for neighbors of vertex, compute edge length
        for (int vv = 0; vv < 3; ++vv)
            {
            vertex2 = h_vn.data[3*vertex1+vv];
            //only look at each pair once
            if(vertex1 < vertex2)
                {
                v2 = h_v.data[vertex2];
                Box->minDist(v1,v2,edge);
                if(norm(edge) < T1Threshold)
                    {
                    bool growCellVertexList = false;
                    getCellVertexSetForT1(vertex1,vertex2,cellSet,vertexSet,growCellVertexList);
                    //forbid a T1 transition that would shrink a triangular cell
                    if( h_cvn.data[cellSet.x] == 3 || h_cvn.data[cellSet.z] == 3)
                        continue;
                    //Does the cell-vertex-neighbor data structure need to be bigger?
                    if(growCellVertexList)
                        {
                        vMax +=1;
                        growCellVerticesList(vMax);
                        h_cv = ArrayHandle<int>(cellVertices,access_location::host,access_mode::readwrite);
                        };

                    //Rotate the vertices in the edge and set them at twice their original distance
                    Dscalar2 midpoint;
                    midpoint.x = v2.x + 0.5*edge.x;
                    midpoint.y = v2.y + 0.5*edge.y;

                    v1.x = midpoint.x-edge.y;
                    v1.y = midpoint.y+edge.x;
                    v2.x = midpoint.x+edge.y;
                    v2.y = midpoint.y-edge.x;
                    Box->putInBoxReal(v1);
                    Box->putInBoxReal(v2);
                    h_v.data[vertex1] = v1;
                    h_v.data[vertex2] = v2;

                    //re-wire the cells and vertices
                    //start with the vertex-vertex and vertex-cell  neighbors
                    for (int vert = 0; vert < 3; ++vert)
                        {
                        //vertex-cell neighbors
                        if(h_vcn.data[3*vertex1+vert] == cellSet.z)
                            h_vcn.data[3*vertex1+vert] = cellSet.w;
                        if(h_vcn.data[3*vertex2+vert] == cellSet.x)
                            h_vcn.data[3*vertex2+vert] = cellSet.y;
                        //vertex-vertex neighbors
                        if(h_vn.data[3*vertexSet.y+vert] == vertex1)
                            h_vn.data[3*vertexSet.y+vert] = vertex2;
                        if(h_vn.data[3*vertexSet.z+vert] == vertex2)
                            h_vn.data[3*vertexSet.z+vert] = vertex1;
                        if(h_vn.data[3*vertex1+vert] == vertexSet.y)
                            h_vn.data[3*vertex1+vert] = vertexSet.z;
                        if(h_vn.data[3*vertex2+vert] == vertexSet.z)
                            h_vn.data[3*vertex2+vert] = vertexSet.y;
                        };
                    //now rewire the cells
                    //cell i loses v2 as a neighbor
                    int cneigh = h_cvn.data[cellSet.x];
                    int cidx = 0;
                    for (int cc = 0; cc < cneigh-1; ++cc)
                        {
                        if(h_cv.data[cellNeighborIndexer(cc,cellSet.x)] == vertex2)
                            cidx +=1;
                        h_cv.data[cellNeighborIndexer(cc,cellSet.x)] = h_cv.data[cellNeighborIndexer(cidx,cellSet.x)];
                        cidx +=1;
                        };
                    h_cvn.data[cellSet.x] -= 1;

                    //cell j gains v2 in between v1 and b
                    cneigh = h_cvn.data[cellSet.y];
                    vector<int> cvcopy1(cneigh+1);
                    cidx = 0;
                    for (int cc = 0; cc < cneigh; ++cc)
                        {
                        int cellIndex = h_cv.data[cellNeighborIndexer(cc,cellSet.y)];
                        cvcopy1[cidx] = cellIndex;
                        cidx +=1;
                        if(cellIndex == vertex1)
                            {
                            cvcopy1[cidx] = vertex2;
                            cidx +=1;
                            };
                        };
                    for (int cc = 0; cc < cneigh+1; ++cc)
                        h_cv.data[cellNeighborIndexer(cc,cellSet.y)] = cvcopy1[cc];
                    h_cvn.data[cellSet.y] += 1;

                    //cell k loses v1 as a neighbor
                    cneigh = h_cvn.data[cellSet.z];
                    cidx = 0;
                    for (int cc = 0; cc < cneigh-1; ++cc)
                        {
                        if(h_cv.data[cellNeighborIndexer(cc,cellSet.z)] == vertex1)
                            cidx +=1;
                        h_cv.data[cellNeighborIndexer(cc,cellSet.z)] = h_cv.data[cellNeighborIndexer(cidx,cellSet.z)];
                        cidx +=1;
                        };
                    h_cvn.data[cellSet.z] -= 1;

                    //cell l gains v1 in between v2 and a
                    cneigh = h_cvn.data[cellSet.w];
                    vector<int> cvcopy2(cneigh+1);
                    cidx = 0;
                    for (int cc = 0; cc < cneigh; ++cc)
                        {
                        int cellIndex = h_cv.data[cellNeighborIndexer(cc,cellSet.w)];
                        cvcopy2[cidx] = cellIndex;
                        cidx +=1;
                        if(cellIndex == vertex2)
                            {
                            cvcopy2[cidx] = vertex1;
                            cidx +=1;
                            };
                        };
                    for (int cc = 0; cc < cneigh+1; ++cc)
                        h_cv.data[cellNeighborIndexer(cc,cellSet.w)] = cvcopy2[cc];
                    h_cvn.data[cellSet.w] = cneigh + 1;

                    };//end condition that a T1 transition should occur
                };
            };//end loop over vertex2
        };//end loop over vertices
    };

/*!
perform whatever check is desired for T1 transtions (here just a "is the edge too short")
and detect whether the edge needs to grow. If so, grow it!
*/
void vertexModelBase::testEdgesForT1GPU()
    {
        {//provide scope for array handles
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
        ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::read);
        ArrayHandle<int> d_vflip(vertexEdgeFlips,access_location::device,access_mode::overwrite);
        ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
        ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::read);
        ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
        ArrayHandle<int> d_grow(growCellVertexListAssist,access_location::device,access_mode::readwrite);

        //first, test every edge, and check if the cellVertices list needs to be grown
        gpu_vm_test_edges_for_T1(d_v.data,
                              d_vn.data,
                              d_vflip.data,
                              d_vcn.data,
                              d_cvn.data,
                              d_cv.data,
                              *(Box),
                              T1Threshold,
                              Nvertices,
                              vertexMax,
                              d_grow.data,
                              cellNeighborIndexer);
        }
    ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::readwrite);
    if(h_grow.data[0] ==1)
        {
        h_grow.data[0]=0;
        growCellVerticesList(vertexMax+1);
        };
    };

/*!
  Iterate through the vertexEdgeFlips list, selecting at most one T1 transition per cell to be done
  on each iteration, until all necessary T1 events have bee performed.
 */
void vertexModelBase::flipEdgesGPU()
    {
    bool keepFlipping = true;
    //By construction, this loop must always run at least twice...save one of the memory transfers
    int iterations = 0;
    while(keepFlipping)
        {
            {//provide scope for ArrayHandles in the multiple-flip-parsing stage
            ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflip(vertexEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflipcur(vertexEdgeFlipsCurrent,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ffe(finishedFlippingEdges,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ef(cellEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int4> d_cs(cellSets,access_location::device,access_mode::readwrite);

            gpu_zero_array(d_ef.data,Ncells);

            gpu_vm_parse_multiple_flips(d_vflip.data,
                               d_vflipcur.data,
                               d_vn.data,
                               d_vcn.data,
                               d_cvn.data,
                               d_cv.data,
                               d_ffe.data,
                               d_ef.data,
                               d_cs.data,
                               cellNeighborIndexer,
                               Ncells);
            };
        //do we need to flip edges? Loop additional times?
        ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::readwrite);
        if(h_ffe.data[0] != 0)
            {
            ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflipcur(vertexEdgeFlipsCurrent,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ef(cellEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int4> d_cs(cellSets,access_location::device,access_mode::readwrite);

            gpu_vm_flip_edges(d_vflipcur.data,
                               d_v.data,
                               d_vn.data,
                               d_vcn.data,
                               d_cvn.data,
                               d_cv.data,
                               d_ef.data,
                               d_cs.data,
                               *(Box),
                               cellNeighborIndexer,
                               Nvertices,
                               Ncells);
            iterations += 1;
            };
        if(h_ffe.data[1]==0)
            keepFlipping = false;

        h_ffe.data[0]=0;
        h_ffe.data[1]=0;
        };//end while loop
    };

/*!
Because the cellVertexList might need to grow, it's convenient to break this into two parts
*/
void vertexModelBase::testAndPerformT1TransitionsGPU()
    {
    testEdgesForT1GPU();
    flipEdgesGPU();
    };

/*!
Trigger a cell death event. This REQUIRES that the vertex model cell to die be a triangle (i.e., we
are mimicking a T2 transition)
*/
void vertexModelBase::cellDeath(int cellIndex)
    {
    //first, throw an error if function is called inappropriately
        {
    ArrayHandle<int> h_cvn(cellVertexNum);
    if (h_cvn.data[cellIndex] != 3)
        {
        printf("Error in vertexModelBase::cellDeath... you are trying to perfrom a T2 transition on a cell which is not a triangle\n");
        throw std::exception();
        };
        }
    //Our strategy will be to completely re-wire everything, and then get rid of the dead entries
    //get the cell and vertex identities of the triangular cell and the cell neighbors
    vector<int> cells(3);
    //For convenience, we will rotate the elements of "vertices" so that the smallest integer is first
    vector<int> vertices(3);
    //also get the vertex neighbors of the vertices (that aren't already part of "vertices")
    vector<int> newVertexNeighbors;
    //So, first create a scope for array handles to write in the re-wired connections
        {//scope for array handle
    ArrayHandle<int> h_cv(cellVertices);
    ArrayHandle<int> h_cvn(cellVertexNum);
    ArrayHandle<int> h_vcn(vertexCellNeighbors);
    int cellsNum=0;
    int smallestV = Nvertices + 1;
    int smallestVIndex = 0;
    for (int vv = 0; vv < 3; ++vv)
        {
        int vIndex = h_cv.data[cellNeighborIndexer(vv,cellIndex)];
        vertices[vv] = vIndex;
        if(vIndex < smallestV)
            {
            smallestV = vIndex;
            smallestVIndex = vv;
            };
        for (int cc =0; cc <3; ++cc)
            {
            int newCell = h_vcn.data[3*vertices[vv]+cc];
            if (newCell == cellIndex) continue;
            bool alreadyFound = false;
            if(cellsNum > 0)
                for (int c2 = 0; c2 < cellsNum; ++c2)
                    if (newCell == cells[c2]) alreadyFound = true;
            if (!alreadyFound)
                {
                cells[cellsNum] = newCell;
                cellsNum +=1;
                }
            };
        };
    std::rotate(vertices.begin(),vertices.begin()+smallestVIndex,vertices.end());
    ArrayHandle<int> h_vn(vertexNeighbors);
    //let's find the vertices connected to the three vertices that form the dying cell
    for (int vv = 0; vv < 3; ++vv)
        {
        for (int v2 = 0; v2 < 3; ++v2)
            {
            int testVertex = h_vn.data[3*vertices[vv]+v2];
            if(testVertex != vertices[0] && testVertex != vertices[1] && testVertex != vertices[2])
                newVertexNeighbors.push_back(testVertex);
            };
        };
    removeDuplicateVectorElements(newVertexNeighbors);
    if(newVertexNeighbors.size() != 3)
        {
        printf("\nError in cell death. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    //Eventually, put the new vertex in, say, the centroid... for now, just put it on top of v1
    Dscalar2 newVertexPosition;
    ArrayHandle<Dscalar2> h_v(vertexPositions);
    newVertexPosition = h_v.data[vertices[0]];

    //First, we start updating the data structures
    //new position of the remaining vertex
    h_v.data[vertices[0]] = newVertexPosition;

    //cell vertices and cell vertex number
    for (int oldCell = 0; oldCell < 3; ++oldCell)
        {
        int cIdx = cells[oldCell];
        int neigh = h_cvn.data[cIdx];
        //fun solution: if the cell includes either v2 or v2, replace with v1 and delete duplicates
        vector<int> vNeighs(neigh);
        for (int vv = 0; vv < neigh; ++vv)
            {
            int vIdx = h_cv.data[cellNeighborIndexer(vv,cIdx)];
            if (vIdx == vertices[1] || vIdx==vertices[2])
                vNeighs[vv] = vertices[0];
            else
                vNeighs[vv] = vIdx;
            };
        removeDuplicateVectorElements(vNeighs);
        h_cvn.data[cIdx] = vNeighs.size();
        for (int vv = 0; vv < vNeighs.size(); ++vv)
            h_cv.data[cellNeighborIndexer(vv,cIdx)] = vNeighs[vv];
        };

    //vertex-vertex and vertex-cell neighbors
    for (int ii = 0; ii < 3; ++ii)
        {
        h_vcn.data[3*vertices[0]+ii] = cells[ii];
        h_vcn.data[3*vertices[1]+ii] = cells[ii];
        h_vcn.data[3*vertices[2]+ii] = cells[ii];
        h_vn.data[3*vertices[0]+ii] = newVertexNeighbors[ii];
        for (int vv = 0; vv < 3; ++vv)
            {
            if (h_vn.data[3*newVertexNeighbors[ii]+vv] == vertices[1] ||
                    h_vn.data[3*newVertexNeighbors[ii]+vv] == vertices[2])
                h_vn.data[3*newVertexNeighbors[ii]+vv] = vertices[0];
            };
        };

    //finally (gross), we need to comb through the data arrays and decrement cell indices greater than cellIdx
    //along with vertex numbers greater than v1 and/or v2
    int v1 = std::min(vertices[1],vertices[2]);
    int v2 = std::max(vertices[1],vertices[2]);
    for (int cv = 0; cv < cellVertices.getNumElements(); ++cv)
        {
        int cellVert = h_cv.data[cv];
        if (cellVert >= v1)
            {
            cellVert -= 1;
            if (cellVert >=v2) cellVert -=1;
            h_cv.data[cv] = cellVert;
            }
        };
    for (int vv = 0; vv < vertexNeighbors.getNumElements(); ++vv)
        {
        int vIdx = h_vn.data[vv];
        if (vIdx >= v1)
            {
            vIdx = vIdx - 1;
            if (vIdx >= v2) vIdx = vIdx - 1;
            h_vn.data[vv] = vIdx;
            };
        };
    for (int vv = 0; vv < vertexCellNeighbors.getNumElements(); ++vv)
        {
        int cIdx = h_vcn.data[vv];
        if (cIdx >= cellIndex)
            h_vcn.data[vv] = cIdx - 1;
        };

        };//scope for array handle... now we get to delete choice array elements

    //Now that the GPUArrays have updated data, let's delete elements from the GPUArrays
    vector<int> vpDeletions = {vertices[1],vertices[2]};
    vector<int> vnDeletions = {3*vertices[1],3*vertices[1]+1,3*vertices[1]+2,
                               3*vertices[2],3*vertices[2]+1,3*vertices[2]+2};
    vector<int> cvDeletions(vertexMax);
    for (int ii = 0; ii < vertexMax; ++ii)
        cvDeletions[ii] = cellNeighborIndexer(ii,cellIndex);
    removeGPUArrayElement(vertexPositions,vpDeletions);
    removeGPUArrayElement(vertexMasses,vpDeletions);
    removeGPUArrayElement(vertexVelocities,vpDeletions);
    removeGPUArrayElement(vertexNeighbors,vnDeletions);
    removeGPUArrayElement(vertexCellNeighbors,vnDeletions);
    removeGPUArrayElement(cellVertexNum,cellIndex);
    removeGPUArrayElement(cellVertices,cvDeletions);

    Nvertices -= 2;
    //phenomenal... let's handle the tag-to-index structures
    ittVertex.resize(Nvertices);
    ttiVertex.resize(Nvertices);
    vector<int> newTagToIdxV(Nvertices);
    vector<int> newIdxToTagV(Nvertices);
    int loopIndex = 0;
    int v1 = std::min(vertices[1],vertices[2]);
    int v2 = std::max(vertices[1],vertices[2]);
    for (int ii = 0; ii < Nvertices+2;++ii)
        {
        int vIdx = tagToIdxVertex[ii]; //vIdx is the current position of the vertex that was originally ii
        if (vIdx != v1 && vIdx != v2)
            {
            if (vIdx >= v1) vIdx = vIdx - 1;
            if (vIdx >= v2) vIdx = vIdx - 1;
            newTagToIdxV[loopIndex] = vIdx;
            loopIndex +=1;
            };
        };
    for (int ii = 0; ii < Nvertices; ++ii)
        newIdxToTagV[newTagToIdxV[ii]] = ii;
    tagToIdxVertex = newTagToIdxV;
    idxToTagVertex = newIdxToTagV;

    //finally, resize remaining stuff and call parent functions
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);

    initializeEdgeFlipLists(); //function call takes care of EdgeFlips and EdgeFlipsCurrent
    Simple2DActiveCell::cellDeath(cellIndex); //This call decrements Ncells by one
    cellNeighborIndexer = Index2D(vertexMax,Ncells);
    };

/*!
Trigger a cell division event, which involves some laborious re-indexing of various data structures.
This simple version of cell division will take a cell and two specified vertices. The edges emanating
clockwise from each of the two vertices will gain a new vertex in the middle of those edges. A new cell is formed by connecting those two new vertices together.
The vector of "parameters" here should be three integers:
parameters[0] = the index of the cell to undergo a division event
parameters[1] = the first vertex to gain a new (clockwise) vertex neighbor.
parameters[2] = the second .....
The two vertex numbers should be between 0 and celLVertexNum[parameters[0]], respectively, NOT the
indices of the vertices being targeted
Note that dParams does nothing
\post This function is meant to be called before the start of a new timestep. It should be immediately followed by a computeGeometry call
*/
void vertexModelBase::cellDivision(const vector<int> &parameters, const vector<Dscalar> &dParams)
    {
    //This function will first do some analysis to identify the cells and vertices involved
    //it will then call base class' cellDivision routine, and then update all needed data structures
    int cellIdx = parameters[0];
    if(cellIdx >= Ncells)
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    int v1 = min(parameters[1],parameters[2]);
    int v2 = max(parameters[1],parameters[2]);

    Dscalar2 cellPos;
    Dscalar2 newV1Pos,newV2Pos;
    int v1idx, v2idx, v1NextIdx, v2NextIdx;
    int newV1CellNeighbor, newV2CellNeighbor;
    bool increaseVertexMax = false;
    int neighs;
    vector<int> combinedVertices;
    {//scope for array handles
    ArrayHandle<Dscalar2> vP(vertexPositions);
    ArrayHandle<int> cellVertNum(cellVertexNum);
    ArrayHandle<int> cv(cellVertices);
    ArrayHandle<int> vcn(vertexCellNeighbors);
    neighs = cellVertNum.data[cellIdx];

    combinedVertices.reserve(neighs+2);
    for (int i = 0; i < neighs; ++i)
        combinedVertices.push_back(cv.data[cellNeighborIndexer(i,cellIdx)]);
    combinedVertices.insert(combinedVertices.begin()+1+v1,Nvertices);
    combinedVertices.insert(combinedVertices.begin()+2+v2,Nvertices+1);

    if(v1 >= neighs || v2 >=neighs)
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    v1idx = cv.data[cellNeighborIndexer(v1,cellIdx)];
    v2idx = cv.data[cellNeighborIndexer(v2,cellIdx)];
    if (v1 < neighs - 1)
        v1NextIdx = cv.data[cellNeighborIndexer(v1+1,cellIdx)];
    else
        v1NextIdx = cv.data[cellNeighborIndexer(0,cellIdx)];
    if (v2 < neighs - 1)
        v2NextIdx = cv.data[cellNeighborIndexer(v2+1,cellIdx)];
    else
        v2NextIdx = cv.data[cellNeighborIndexer(0,cellIdx)];

    //find the positions of the new vertices
    Dscalar2 disp;
    Box->minDist(vP.data[v1NextIdx],vP.data[v1idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV1Pos = vP.data[v1idx] + disp;
    Box->putInBoxReal(newV1Pos);
    Box->minDist(vP.data[v2NextIdx],vP.data[v2idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV2Pos = vP.data[v2idx] + disp;
    Box->putInBoxReal(newV2Pos);

    //find the third cell neighbor of the new vertices
    int ans = -1;
    for (int vi = 3*v1idx; vi < 3*v1idx+3; ++vi)
        for (int vj = 3*v1NextIdx; vj < 3*v1NextIdx+3; ++vj)
            {
            int c1 = vcn.data[vi];
            int c2 = vcn.data[vj];
            if ((c1 == c2) &&(c1 != cellIdx))
                ans = c1;
            };
    if (ans >=0)
        newV1CellNeighbor = ans;
    else
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    ans = -1;
    for (int vi = 3*v2idx; vi < 3*v2idx+3; ++vi)
        for (int vj = 3*v2NextIdx; vj < 3*v2NextIdx+3; ++vj)
            {
            int c1 = vcn.data[vi];
            int c2 = vcn.data[vj];
            if ((c1 == c2) &&(c1 != cellIdx))
                ans = c1;
            };
    if (ans >=0)
        newV2CellNeighbor = ans;
    else
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    if(cellVertNum.data[newV1CellNeighbor] + 1 >=vertexMax)
        increaseVertexMax = true;
    if(cellVertNum.data[newV2CellNeighbor] + 1 >=vertexMax)
        increaseVertexMax = true;
    }//end scope of old array handles... new vertices and cells identified

    //update cell and vertex number; have access to both new and old indexer if vertexMax changes
    Index2D n_idxOld(vertexMax,Ncells);
    if (increaseVertexMax)
        {
        printf("vertexMax has increased due to cell division\n");
        vertexMax += 2;
        };

    //The Simple2DActiveCell routine will update Motility and cellDirectors,
    // it in turn calls the Simple2DCell routine, which grows its data structures and increment Ncells by one
    Simple2DActiveCell::cellDivision(parameters);

    Nvertices += 2;

    //additions to the spatial sorting vectors...
    ittVertex.push_back(Nvertices-2); ittVertex.push_back(Nvertices-1);
    ttiVertex.push_back(Nvertices-2); ttiVertex.push_back(Nvertices-1);
    tagToIdxVertex.push_back(Nvertices-2); tagToIdxVertex.push_back(Nvertices-1);
    idxToTagVertex.push_back(Nvertices-2); idxToTagVertex.push_back(Nvertices-1);

    //GPUArrays that just need their length changed
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    initializeEdgeFlipLists(); //function call takes care of EdgeFlips and EdgeFlipsCurrent
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);

    //use the copy and grow mechanism where we need to actually set values
    growGPUArray(vertexPositions,2); //(nv)
    growGPUArray(vertexMasses,2); //(nv)
    growGPUArray(vertexVelocities,2); //(nv)
    growGPUArray(vertexNeighbors,6); //(3*nv)
    growGPUArray(vertexCellNeighbors,6); //(3*nv)
    growGPUArray(cellVertexNum,1); //(nc)
    //the index cellVertices array needs more care...
    vector<int>  cellVerticesVec;
    copyGPUArrayData(cellVertices,cellVerticesVec);
    cellVertices.resize(vertexMax*Ncells);
    //first, let's take care of the vertex positions, masses, and velocities
        {//arrayhandle scope
        ArrayHandle<Dscalar2> h_vp(vertexPositions);
        h_vp.data[Nvertices-2] = newV1Pos;
        h_vp.data[Nvertices-1] = newV2Pos;
        ArrayHandle<Dscalar2> h_vv(vertexVelocities);
        h_vv.data[Nvertices-2] = make_Dscalar2(0.0,0.0);
        h_vv.data[Nvertices-1] = make_Dscalar2(0.0,0.0);
        ArrayHandle<Dscalar> h_vm(vertexMasses);
        h_vm.data[Nvertices-2] = h_vm.data[v1idx];
        h_vm.data[Nvertices-1] = h_vm.data[v2idx];
        }

    //the vertex-vertex neighbors
        {//arrayHandle scope
        ArrayHandle<int> h_vv(vertexNeighbors);
        //new v1
        h_vv.data[3*(Nvertices-2)+0] = v1idx;
        h_vv.data[3*(Nvertices-2)+1] = v1NextIdx;
        h_vv.data[3*(Nvertices-2)+2] = Nvertices-1;
        //new v2
        h_vv.data[3*(Nvertices-1)+0] = Nvertices-2;
        h_vv.data[3*(Nvertices-1)+1] = v2idx;
        h_vv.data[3*(Nvertices-1)+2] = v2NextIdx;
        //v1idx
        for (int ii = 3*v1idx; ii < 3*(v1idx+1); ++ii)
            if (h_vv.data[ii] == v1NextIdx) h_vv.data[ii] = Nvertices-2;
        //v1NextIdx
        for (int ii = 3*v1NextIdx; ii < 3*(v1NextIdx+1); ++ii)
            if (h_vv.data[ii] == v1idx) h_vv.data[ii] = Nvertices-2;
        //v2idx
        for (int ii = 3*v2idx; ii < 3*(v2idx+1); ++ii)
            if (h_vv.data[ii] == v2NextIdx) h_vv.data[ii] = Nvertices-1;
        //v2NextIdx
        for (int ii = 3*v2NextIdx; ii < 3*(v2NextIdx+1); ++ii)
            if (h_vv.data[ii] == v2idx) h_vv.data[ii] = Nvertices-1;
        };

    //for computing vertex-cell neighbors and cellVertices, recall that combinedVertices is a list:
    //v0, v1.. newvertex 1... newvertex2 ... v_old_last_vertex
    //for convenience, rotate this so that it is newvertex1 ... newvertex2, (other vertices), and
    //create another vector that is newvertex2...newvertex1, (other vertices)
    vector<int> cv2=combinedVertices;
    rotate(cv2.begin(), cv2.begin()+v2+2, cv2.end());
    rotate(combinedVertices.begin(), combinedVertices.begin()+v1+1, combinedVertices.end());
    int nVertNewCell = (v2 - v1) +2;
    int nVertCellI = neighs+2-(v2-v1);
        {//arrayHandle scope
        ArrayHandle<int> h_cvn(cellVertexNum);
        h_cvn.data[Ncells-1] = nVertNewCell;
        h_cvn.data[cellIdx] = nVertCellI;

        ArrayHandle<int> h_vcn(vertexCellNeighbors);
        //new v1
        h_vcn.data[3*(Nvertices-2)+0] = newV1CellNeighbor;
        h_vcn.data[3*(Nvertices-2)+1] = Ncells-1;
        h_vcn.data[3*(Nvertices-2)+2] = cellIdx;
        //new v2
        h_vcn.data[3*(Nvertices-1)+0] = newV2CellNeighbor;
        h_vcn.data[3*(Nvertices-1)+1] = cellIdx;
        h_vcn.data[3*(Nvertices-1)+2] = Ncells-1;
        //vertices in between newV1 and newV2 don't neighbor the divided cell any more
        for (int i = 1; i < nVertNewCell-1; ++i)
            for (int vv = 0; vv < 3; ++vv)
                if(h_vcn.data[3*combinedVertices[i]+vv] == cellIdx)
                    h_vcn.data[3*combinedVertices[i]+vv] = Ncells-1;
        };

    // finally, reset the vertices associated with every cell
        {//arrayHandle scope
        ArrayHandle<int> cv(cellVertices);
        ArrayHandle<int> h_cvn(cellVertexNum);
        //first, copy over the old cells with any new indexing
        for (int cell = 0; cell < Ncells -1; ++cell)
            {
            int ns = h_cvn.data[cell];
            for (int vv = 0; vv < ns; ++vv)
                cv.data[cellNeighborIndexer(vv,cell)] = cellVerticesVec[n_idxOld(vv,cell)];
            };
        //correct cellIdx's vertices
        for (int vv = 0; vv < nVertCellI; ++vv)
            cv.data[cellNeighborIndexer(vv,cellIdx)] = cv2[vv];
        //add the vertices to the new cell
        for (int vv = 0; vv < nVertNewCell; ++vv)
            cv.data[cellNeighborIndexer(vv,Ncells-1)] = combinedVertices[vv];

        //insert the vertices into newV1CellNeighbor and newV2CellNeighbor
        vector<int> cn1, cn2;
        int cn1Size = h_cvn.data[newV1CellNeighbor];
        int cn2Size = h_cvn.data[newV2CellNeighbor];
        cn1.reserve(cn1Size+1);
        cn2.reserve(cn2Size+1);
        for (int i = 0; i < cn1Size; ++i)
            {
            int curVertex = cv.data[cellNeighborIndexer(i,newV1CellNeighbor)];
            cn1.push_back(curVertex);
            if(curVertex == v1NextIdx)
                cn1.push_back(Nvertices-2);
            };
        for (int i = 0; i < cn2Size; ++i)
            {
            int curVertex = cv.data[cellNeighborIndexer(i,newV2CellNeighbor)];
            cn2.push_back(curVertex);
            if(curVertex == v2NextIdx)
                cn2.push_back(Nvertices-1);
            };

        //correct newV1CellNeighbor's vertices
        for (int vv = 0; vv < cn1Size+1; ++vv)
            cv.data[cellNeighborIndexer(vv,newV1CellNeighbor)] = cn1[vv];
        //correct newV2CellNeighbor's vertices
        for (int vv = 0; vv < cn2Size+1; ++vv)
            cv.data[cellNeighborIndexer(vv,newV2CellNeighbor)] = cn2[vv];
        //correct the number of vertex neighbors of the cells
        h_cvn.data[newV1CellNeighbor] = cn1Size+1;
        h_cvn.data[newV2CellNeighbor] = cn2Size+1;
        };
    };
