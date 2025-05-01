//
// Starting point for the OpenCL coursework for COMP/XJCO3221 Parallel Computation.
//
// Once compiled, execute with the size of the square grid as a command line argument, i.e.
//
// ./cwk3 16
//
// will generate a 16 by 16 grid. The C-code below will then display the initial grid,
// followed by the same grid again. You will need to implement OpenCL that applies the heat
// equation as per the instructions, so that the final grid is different.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 2 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// getCmdLineArg()  :  Parses grid size N from command line argument, or fails with error message.
// fillGrid()       :  Fills the grid with random values, except boundary values which are always zero.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
 
    //
    // Parse command line argument and check it is valid. Handled by a routine in the helper file.
    //
    int N;
    getCmdLineArg( argc, argv, &N );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the grid. For simplicity, this uses a one-dimensional array. On Host.
	float *hostGrid = (float*) malloc( N * N * sizeof(float) );

	// Fill the grid with some initial values, and display to stdout. fillGrid() is defined in the helper file.
    fillGrid( hostGrid, N );
    printf( "Original grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

	//
	// Allocate memory for the grid(s) on the GPU and apply the heat equation as per the instructions.
	//

    // allocates memory for the orig and new grid on gpu
    cl_mem device_grid_original = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N*N*sizeof(float),
        hostGrid,
        &status
    );
    cl_mem device_grid_new = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        N*N*sizeof(float),
        NULL,
        &status
    );

    // compile file
    cl_kernel kernel = compileKernelFromFile(
        "cwk3.cl",
        "computeCell",
        context,
        device
    );


    // set arguments
    status = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &device_grid_original
    );

    status = clSetKernelArg(
        kernel,
        1,
        sizeof(cl_mem),
        &device_grid_new
    );

    status = clSetKernelArg(
        kernel,
        2,
        sizeof(int),
        &N
    );

    // set group and index space size

    size_t maxWorkItems ;
    clGetDeviceInfo( device , CL_DEVICE_MAX_WORK_GROUP_SIZE ,
                    sizeof ( size_t ) ,& maxWorkItems , NULL );
    
    // must satisfy: x <= n; x | N; x^2 <= maxWorkItems;
    int x = N;
    while (x > 0) {
        if (x * x <= maxWorkItems) {
            break; 
        }
        x = x>>1;
    };

    size_t globalWorkSize[2] = {N, N};
    size_t workGroupSize[2] = {x, x};

    // start the kernel
    status = clEnqueueNDRangeKernel(
        queue,
        kernel,
        2,
        NULL,
        globalWorkSize,
        workGroupSize,
        0,
        NULL,
        NULL
    );

    // get result back to host
    status = clEnqueueReadBuffer(
        queue,
        device_grid_new,
        CL_TRUE,
        0,
        N*N*sizeof(float),
        hostGrid,
        0, 
        NULL, 
        NULL
    );


    //
    // Display the final result. This assumes that the iterated grid was copied back to the hostGrid array.
    //
    printf( "Final grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );
    clReleaseMemObject(device_grid_original);
    clReleaseMemObject(device_grid_new);
    clReleaseKernel(kernel);

    free( hostGrid );

    return EXIT_SUCCESS;
}

