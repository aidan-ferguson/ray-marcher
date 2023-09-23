#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>
#include <thread>

#include <glm/glm.hpp>

// Right-handed coordinate system:
//  +X -> West
//  +Y -> Up
//  +Z -> North

// General options
const int PROGRESS_BAR_WIDTH = 100;

// Rendering options
const int WIDTH = 800;
const int HEIGHT = 600;
const float H_FOV = 90.0f * (M_PI / 180);
const float V_FOV = 2 * std::atan(std::tan(H_FOV/2.0f)*((float)HEIGHT/(float)WIDTH));
const int N_SAMPLES = 8;
const unsigned int MAX_THREADS = 40;
const unsigned int N_FRAMES = 350;
const unsigned int MS_PER_FRAME = 20; // Number of (in-sim) milliseconds per frame

// Random stuff used for super-sampling
std::random_device rd;
std::mt19937 generator(rd());
std::uniform_real_distribution<> random_dist(-0.5, 0.5);

// Camera
const glm::vec3 CAM_POS = glm::vec3(0, 0, 3);
const glm::vec3 SCREEN_TOP_LEFT = glm::vec3(-std::tan(0.5f * H_FOV), -std::tan(0.5f * V_FOV), -1);
const glm::vec3 SCREEN_BOTTOM_RIGHT = glm::vec3(std::tan(0.5f * H_FOV), std::tan(0.5f * V_FOV), -1);

// Ray marching parameters
const int MAX_RAY_STEPS = 100;
const float STEP_EPSILON_THRESHOLD = 0.001f;

// Get the minimum distance to the scene given a point in space
float min_dist_to_scene(glm::vec3 pos) {
    // Mandlebulb SDF from http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
    glm::vec3 z = pos;
    int Iterations = 10;
    int Bailout = 100.0f;
    float Power = 8.0f;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < Iterations ; i++) {
		r = length(z);
		if (r>Bailout) break;
		
		// convert to polar coordinates
		float theta = acos(z.z/r);
		float phi = glm::atan(z.y,z.x);
		dr =  pow( r, Power-1.0)*Power*dr + 1.0;
		
		// scale and rotate the point
		float zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;
		
		// convert back to cartesian coordinates
		z = zr*glm::vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		z+=pos;
	}
	return 0.5*log(r)*r/dr;
}

// Take a normalised ray direction and return the number of steps the ray took into the scene
int march_ray(glm::vec3 ray_dir, float time){
    glm::vec3 ray_pos = CAM_POS;

    // Rotate the camera around the Y axis with time
    glm::mat3 pos_rot_mat = {
        {std::cos(time) , 0, std::sin(time)},
        {0              , 1, 0},
        {-std::sin(time), 0, std::cos(time)}
    };
    ray_pos = pos_rot_mat * ray_pos;
    ray_dir = pos_rot_mat * ray_dir;

    int n_steps = 0;
    for(int step = 0; step < MAX_RAY_STEPS; step++) {
        // Step forward maximum allowed amount
        float min_dist = min_dist_to_scene(ray_pos);
        if(min_dist < STEP_EPSILON_THRESHOLD) 
            return n_steps;
        ray_pos += ray_dir*min_dist;
        n_steps++;
    }

    return n_steps;
}

glm::vec3 get_pixel(int x, int y, float time) {
    // Random super-sampling for each pixel
    glm::vec3 pixel = glm::vec3(0,0,0);
    for(int sample = 0; sample < N_SAMPLES; sample++) {
        // Calculate ray dir +/- some randomness for the super-sampling
        float x_percentage = ((float)x+(float)random_dist(generator))/(float)WIDTH;
        float y_percentage = ((float)y+(float)random_dist(generator))/(float)HEIGHT;
        
        // Get ray direction by getting direction from cam to 'screen' in-front of camera
        float screen_x = SCREEN_TOP_LEFT.x + (SCREEN_BOTTOM_RIGHT.x - SCREEN_TOP_LEFT.x)*x_percentage;
        float screen_y = SCREEN_TOP_LEFT.y + (SCREEN_BOTTOM_RIGHT.y - SCREEN_TOP_LEFT.y)*y_percentage; 
        glm::vec3 ray_dir = glm::normalize(glm::vec3(screen_x, screen_y, -1));

        // Iteratively step the ray forward and return the steps taken
        pixel += ((float)march_ray(ray_dir, time)/(float)MAX_RAY_STEPS);
    }

    // Average pixel
    pixel /= N_SAMPLES;

    return (glm::vec3(1,1,1) - pixel);
}

void thread_compute_region(glm::vec3** pixels, uint32_t start_y, uint32_t end_y, float time)
{
    // Populate pixels
    for(int y = start_y; y < end_y; y++) {
        for(int x = 0; x < WIDTH; x++) {
            pixels[y][x] = get_pixel(x, y, time);
        }
    }
}

void print_progress(int frame_idx)
{
    float progress = static_cast<float>(frame_idx)/static_cast<float>(N_FRAMES);

    std::string progress_string("Frame: (");
    progress_string += std::to_string(frame_idx) + "/" + std::to_string(N_FRAMES) + ") ";
    std::string filled_bar(static_cast<int>(PROGRESS_BAR_WIDTH*progress), '#');
    progress_string += "[" + filled_bar + std::string(PROGRESS_BAR_WIDTH-filled_bar.size(), '-') + "]";

    printf("%s\r", progress_string.c_str());
    fflush(stdout);
}

int main() {
    // Setup
    const uint32_t n_threads = std::min(std::thread::hardware_concurrency(), MAX_THREADS);
    assert(n_threads != 0);
    glm::vec3** pixels = new glm::vec3*[HEIGHT];
    for(int idx = 0; idx < HEIGHT; idx++) {
        pixels[idx] = new glm::vec3[WIDTH];
    }

    for (int frame_idx = 0; frame_idx < N_FRAMES; frame_idx++)
    {
        print_progress(frame_idx);

        // Calculate timestamp based on frame 
        float time = static_cast<float>(frame_idx * MS_PER_FRAME) / 1000.0f;

        // Start N threads that each populate a portion of the pixels
        std::vector<std::thread> threads;
        const uint32_t height_interval = HEIGHT/n_threads;
        for(int thread_idx = 0; thread_idx < n_threads; thread_idx++)
        {
            uint32_t start_y = static_cast<uint32_t>(height_interval*thread_idx);
            // Special case for the last thread, we want to go all the way to the end of the image
            uint32_t end_y = static_cast<uint32_t>(height_interval*(thread_idx+1));
            if (thread_idx == n_threads-1)
            {
                end_y = HEIGHT;
            }

            // Create lambda for this thread
            threads.push_back(std::thread(thread_compute_region, pixels, start_y, end_y, time));
        }

        // Wait on threads completing
        for(int thread_idx = 0; thread_idx < threads.size(); thread_idx++)
        {
            threads[thread_idx].join();
        }

        // Write PPM file
        std::ofstream file;
        file.open (std::string("frames/") + std::to_string(frame_idx) + std::string(".ppm"));
        file << "P3 " << WIDTH << " " << HEIGHT <<  " 256\n";
        for(int y = 0; y < HEIGHT; y++) {
            for(int x = 0; x < WIDTH; x++) {
                glm::vec3 pixel = pixels[y][x] * 256.0f;
                file << (int)pixel.r << " " << (int)pixel.g << " " << (int)pixel.b << " ";
            }
            file << "\n";
        }
        file.close();
    }

    // No explicit delete[] 's as the OS will cleanup
    return 0;
}