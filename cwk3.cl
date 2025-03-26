// Kernel for the heat equation.
__kernel
void computeCell(__global const float* device_grid_original, __global float* device_grid_new, int N)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int xy = gidy * N + gidx;

    if (gidx == 0 || gidy == 0 || gidx == N - 1 || gidy == N - 1) {
        device_grid_new[xy] = 0.0f;
        return;
    }

    float top    = device_grid_original[(gidy - 1) * N + gidx];
    float bottom = device_grid_original[(gidy + 1) * N + gidx];
    float left   = device_grid_original[gidy * N + (gidx - 1)];
    float right  = device_grid_original[gidy * N + (gidx + 1)];

    device_grid_new[xy] = 0.25f * (top + bottom + left + right);
}
