#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>
#include <thread>

#include <curand_kernel.h>
#include <cuda_runtime.h>

// Right-handed coordinate system:
//  +X -> West
//  +Y -> Up
//  +Z -> North

// General options
const int PROGRESS_BAR_WIDTH = 100;

// Rendering options
const int WIDTH = 1920;
const int HEIGHT = 1080;
const float H_FOV = 90.0f * (M_PI / 180);
const float V_FOV = 2 * std::atan(std::tan(H_FOV/2.0f)*((float)HEIGHT/(float)WIDTH));
const int N_SAMPLES = 10;
const unsigned int N_FRAMES = 350; // 350 @ 20ms per frame -> full rotation
const unsigned int MS_PER_FRAME = 20; // Number of (in-sim) milliseconds per frame
const float DIFFUSE_LIGHTING = 0.3f;
const int MAX_CUDA_THREADS = 10'000'000;

// Camera, values copied in main
__constant__ float CAM_POS[3];
__constant__ float SCREEN_TOP_LEFT[3];
__constant__ float SCREEN_BOTTOM_RIGHT[3];
__constant__ float LIGHT_POS[3];

// Ray marching parameters
const int MAX_RAY_STEPS = 100;
const float STEP_EPSILON_THRESHOLD = 0.001f;

// Get the minimum distance to the scene given a point in space
__device__ float min_dist_to_scene(float pos[3], float time) {
    // Mandlebulb SDF from http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
    float w = sin(time/9.0);
    float z[4] = {pos[0], pos[1], pos[2], w};
    int Iterations = 10;
    float Bailout = 100.0f;
    float Power = 8.0f;
    float dr = 1.0f;
    float r = 0.0f;
    for (int i = 0; i < Iterations; i++) {
        r = sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2] + z[3]*z[3]);
        if (r > Bailout) break;

        // convert to polar coordinates (4D)
        float theta = acos(z[2] / r);
        float phi = atan(z[1] / z[0]);
        float psi = atan(sqrt(z[0] * z[0] + z[1] * z[1]) / z[3]);
        dr = pow(r, Power - 1.0f) * Power * dr + 1.0f;

        // scale and rotate the point
        float zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;
        psi = psi * Power;

        // convert back to cartesian coordinates (4D)
        z[0] = (zr * sin(psi) * cos(theta) * cos(phi)) + pos[0];
        z[1] = (zr * sin(psi) * sin(phi) * cos(theta)) + pos[1];
        z[2] = (zr * cos(psi) * cos(theta)) + pos[2];
        z[3] = (zr * sin(psi) * sin(theta)) + w;
    }
    return 0.5f * log(r) * r / dr;
}

// Take a normalised ray direction and return the number of steps the ray took into the scene
__device__ int march_ray(float ray_pos[3], float ray_dir[3], float time){
    int step = 0;
    for(; step < MAX_RAY_STEPS; step++) {
        // Step forward maximum allowed amount
        float min_dist = min_dist_to_scene(ray_pos, time);
        if(min_dist < STEP_EPSILON_THRESHOLD) 
            return;
        ray_pos[0] += ray_dir[0]*min_dist;
        ray_pos[1] += ray_dir[1]*min_dist;
        ray_pos[2] += ray_dir[2]*min_dist;
    }
    return step;
}

__device__ void mat_vector_dot(float* mat_a, float* vec, int mat_r, int mat_c, int vec_sz)
{
    float vec_cpy[3] = {0};
    for (int idx = 0; idx < vec_sz; idx++)
    {
        vec_cpy[idx] = vec[idx];
    }
    
    for (int i = 0; i < mat_r; i++)
    {   
        float sum = 0.0f;
        for (int j = 0; j < mat_r; j++)
        {
            sum += mat_a[(mat_c*i) + j] * vec_cpy[j];
        }
        vec[i] = sum;
    }
}


__global__ void init_kernel(curandState_t* state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(618, idx, 0, &state[idx]);
}


__global__ void compute_pixel(uint8_t* frame, curandState_t* rand_state, float time) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int x = thread_idx % WIDTH;
    const int y = thread_idx / WIDTH; 

    // TODO: move outside of kernel
    // Rotate the camera around the Y axis with time
    float pos_rot_mat [3*3] = {
        std::cos(0.001f) , 0, std::sin(0.001f),
        0              , 1, 0,
        -std::sin(0.001f), 0, std::cos(0.001f)
    };

    // Random super-sampling for each pixel
    float pixel[3] = {0,0,0};
    for(int sample = 0; sample < N_SAMPLES; sample++) {
        // Calculate ray dir +/- some randomness for the super-sampling
        float x_percentage = ((float)x+(curand_uniform(rand_state+thread_idx)-0.5f))/(float)WIDTH;
        float y_percentage = ((float)y+(curand_uniform(rand_state+thread_idx)-0.5f))/(float)HEIGHT;
        
        // Get ray direction by getting direction from cam to 'screen' in-front of camera
        float screen_x = SCREEN_TOP_LEFT[0] + (SCREEN_BOTTOM_RIGHT[0] - SCREEN_TOP_LEFT[0])*x_percentage;
        float screen_y = SCREEN_TOP_LEFT[1] + (SCREEN_BOTTOM_RIGHT[1] - SCREEN_TOP_LEFT[1])*y_percentage; 

        // Normalise ray_dir
        float ray_dir[3] = {screen_x, screen_y, -1};
        for (int idx = 0; idx < 3; idx++)
        {
            ray_dir[idx] /= std::sqrt(screen_x*screen_x + screen_y*screen_y + 1);
        }

        // Zoom in with time
        float ray_pos[3];
        memcpy(ray_pos, CAM_POS, sizeof(float)*3);
        // ray_pos[2] = ((std::cos(time)+1.0f)/2.0f) + 1.1f;

        // Rotate with time
        mat_vector_dot(pos_rot_mat, ray_pos, 3, 3, 3);
        mat_vector_dot(pos_rot_mat, ray_dir, 3, 3, 3);

        // Iteratively step the ray forward and return the steps taken
        int march_steps = march_ray(ray_pos, ray_dir, time);
        float current_sample = 1.0f; //(static_cast<float>(march_steps)/static_cast<float>(MAX_RAY_STEPS));

        // Now that we have the position of the intersection if it is not MAX_RAY_STEPS, try to ray-march towards the light source, if it reaches MAX_RAY_STEPS
        //  multiply the colour of the pixel with the number of steps taken (as a percentage of MAX_RAY_STEPS)
        if (march_steps < MAX_RAY_STEPS)
        {
            float light_dir[3] = {LIGHT_POS[0] - ray_pos[0], LIGHT_POS[1] - ray_pos[1], LIGHT_POS[2] - ray_pos[2]};
            float light_dir_mag = std::sqrt(light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1] + light_dir[2]*light_dir[2]);
            for (int idx = 0; idx < 3; idx++)
            {
                light_dir[idx] /= light_dir_mag;
            }

            ray_pos[0] += (light_dir[0] * STEP_EPSILON_THRESHOLD);
            ray_pos[1] += (light_dir[1] * STEP_EPSILON_THRESHOLD);
            ray_pos[2] += (light_dir[2] * STEP_EPSILON_THRESHOLD);
            march_steps = march_ray(ray_pos, light_dir, time);

            // Get the number of steps taken towards the light as a percentage of MAX_RAY_STEPS -> then do 1 - that as pixel values are currently flipped (i.e.) 1 = black
            float light_steps = (static_cast<float>(march_steps)/static_cast<float>(MAX_RAY_STEPS));

            current_sample *= light_steps;

            // Add diffuse lighting
            current_sample += DIFFUSE_LIGHTING;
        }
        else
        {
            current_sample = 0.0f;
        }

        pixel[0] += current_sample;
        pixel[1] += current_sample;
        pixel[2] += current_sample;
    }

    // // Average pixel
    pixel[0] /= N_SAMPLES;
    pixel[1] /= N_SAMPLES;
    pixel[2] /= N_SAMPLES;

    uint8_t* p_pixel = &frame[3*(y*WIDTH + x)];
    p_pixel[0] = min(pixel[0], 1.0f) * 255.0f;
    p_pixel[1] = min(pixel[1], 1.0f) * 255.0f;
    p_pixel[2] = min(pixel[2], 1.0f) * 255.0f;
}

void print_progress(int frame_idx)
{
    float progress = static_cast<float>(frame_idx+1)/static_cast<float>(N_FRAMES);

    std::string progress_string("Frame: (");
    progress_string += std::to_string(frame_idx+1) + "/" + std::to_string(N_FRAMES) + ") ";
    std::string filled_bar(static_cast<int>(PROGRESS_BAR_WIDTH*progress), '#');
    progress_string += "[" + filled_bar + std::string(PROGRESS_BAR_WIDTH-filled_bar.size(), '-') + "]";

    printf("%s\r", progress_string.c_str());
    fflush(stdout);
}

int main() {
    // GPU Info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA acceleration on '%s'\n", prop.name);
    printf("Grid Limits: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Setup
    uint8_t* cu_pixels = nullptr;
    uint8_t* host_pixels = new uint8_t[3*WIDTH*HEIGHT];
    cudaMalloc(&cu_pixels, 3*WIDTH*HEIGHT);

    // Copy camera values
    const float cam_pos_temp[3] = {0, 0, 4};
    const float screen_top_left_temp[3] = {-tan(0.5f * H_FOV), -tan(0.5f * V_FOV), -1};
    const float screen_bottom_right_temp[3] = {tan(0.5f * H_FOV), tan(0.5f * V_FOV), -1};
    const float light_pos_temp[3] = {0, -10, 3};
    cudaMemcpyToSymbol(CAM_POS, cam_pos_temp, 3 * sizeof(float));
    cudaMemcpyToSymbol(SCREEN_TOP_LEFT, screen_top_left_temp, 3 * sizeof(float));
    cudaMemcpyToSymbol(SCREEN_BOTTOM_RIGHT, screen_bottom_right_temp, 3 * sizeof(float));
    cudaMemcpyToSymbol(LIGHT_POS, light_pos_temp, 3 * sizeof(float));
                
    // Note, currently too many threads will result in region of image not being computed
    int cuda_threads = std::min(WIDTH*HEIGHT, MAX_CUDA_THREADS);
    int blockSize = 256;
    int gridSize = (cuda_threads + blockSize - 1) / blockSize;
    // std::cout << grid_size << std::endl;

    curandState_t* rand_state;
    cudaMalloc(&rand_state, sizeof(curandState_t) * (gridSize*blockSize));
    init_kernel<<<gridSize,blockSize>>>(rand_state);

    for (int frame_idx = 0; frame_idx < N_FRAMES; frame_idx++)
    {
        print_progress(frame_idx);

        // Calculate timestamp based on frame 
        float time = static_cast<float>(frame_idx * MS_PER_FRAME) / 1000.0f;

        // Run frame compute kernel
        compute_pixel<<<gridSize,blockSize>>>(cu_pixels, rand_state, time);
        cudaMemcpy(host_pixels, cu_pixels, 3*WIDTH*HEIGHT, cudaMemcpyDeviceToHost);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        // Write PPM file
        std::ofstream file;
        file.open (std::string("frames/") + std::to_string(frame_idx) + std::string(".ppm"));
        file << "P3 " << WIDTH << " " << HEIGHT <<  " 256\n";
        for(int y = 0; y < HEIGHT; y++) {
            for(int x = 0; x < WIDTH; x++) {
                uint8_t* pixel = &host_pixels[3*(y*WIDTH + x)];
                file << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << " ";
            }
            file << "\n";
        }
        file.close();
    }

    // Cleanup
    printf("\nComplete.\n");

    cudaFree(cu_pixels);
    delete[] host_pixels;

    return 0;
}