#include <iostream>
#include <glm/glm.hpp>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>

// Rendering options
const int WIDTH = 1920;
const int HEIGHT = 1080;
const float H_FOV = 90.0f * (M_PI / 180);
const float V_FOV = 2 * std::atan(std::tan(H_FOV/2.0f)*((float)HEIGHT/(float)WIDTH));
const int N_SAMPLES = 8;

// Random stuff used for supersampling
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
int march_ray(glm::vec3 ray_dir){
    glm::vec3 ray_pos = CAM_POS;
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

glm::vec3 get_pixel(int x, int y) {
    // Random super-sampling for each pixel
    glm::vec3 pixel = glm::vec3(0,0,0);
    for(int sample = 0; sample < N_SAMPLES; sample++) {
        // Calculate ray dir +/- some randomness for the supersampling
        float x_percentage = ((float)x+(float)random_dist(generator))/(float)WIDTH;
        float y_percentage = ((float)y+(float)random_dist(generator))/(float)HEIGHT;
        
        // Get ray direction by getting direction from cam to 'screen' infront of camera
        float screen_x = SCREEN_TOP_LEFT.x + (SCREEN_BOTTOM_RIGHT.x - SCREEN_TOP_LEFT.x)*x_percentage;
        float screen_y = SCREEN_TOP_LEFT.y + (SCREEN_BOTTOM_RIGHT.y - SCREEN_TOP_LEFT.y)*y_percentage; 
        glm::vec3 ray_dir = glm::normalize(glm::vec3(screen_x, screen_y, -1));
        
        // Iterativly step the ray forward and return the steps taken
        pixel += ((float)march_ray(ray_dir)/(float)MAX_RAY_STEPS);
    }

    // Average pixel
    pixel /= N_SAMPLES;

    return (glm::vec3(1,1,1) - pixel);
}

int main() {
    // Setup
    std::vector<std::vector<glm::vec3>> pixels;
    pixels.resize(HEIGHT);
    for(std::vector<glm::vec3>& row : pixels)
        row.resize(WIDTH);    

    // Populate pixels
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            pixels[y][x] = get_pixel(x, y);
        }
    }

    // Write PPM file
    std::ofstream file;
    file.open ("output.ppm");
    file << "P3 " << WIDTH << " " << HEIGHT <<  " 256\n";
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            glm::vec3 pixel = pixels[y][x] * 256.0f;
            file << (int)pixel.r << " " << (int)pixel.g << " " << (int)pixel.b << " ";
        }
        file << "\n";
    }
    file.close();

    return 0;
}