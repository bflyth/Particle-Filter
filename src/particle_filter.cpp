/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define EPS = .0001; //Very small number
#define NUMBER_OF_PARTICLES = 300;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized) {

		//initialize number of particles
		num_particles = NUMBER_OF_PARTICLES;

		//initialize a rgn with gaussian distribution
		default_random_engine gen;
		
		//generate normal distribution for x, y, theta
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double>	dist_theta(theta, std[2]);

		//resize vector to fit particles
		particles.resize(num_particles);
		weights.resize(num_particles);

		//initialize all particles
		for (int i = 0; i < num_particles; i++) {
			particles[i].id		= i;
			particles[i].x		= dist_x(gen);
			particles[i].y		= dist_y(gen);
			particles[i].theta  = dist_theta(gen);
			particles[i].weight = 1.0;
		}
		is_initialized = true;
		return;
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	//rgn with gaussian distribution
	default_random_engine gen;

	//normal distributions of x, y, and theta
	normal_distribution<double> dist_x(0., std_pos[0]);
	normal_distribution<double> dist_y(0., std_pos[1]);
	normal_distribution<double> dist_theta(0., std_pos[2]);

	//a constant for velocity change
	const double vel_delta = velocity * delta_t;

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) > EPS) {
			//ratio of turn to velocity
			double vel_yaw_ratio = velocity / yaw_rate;
			//change in turn rate
			double yaw_delta = yaw_rate * delta_t;

			//add noise to x, y, and theta
			particles[i].x += vel_yaw_ratio * (sin(particles[i].theta + yaw_delta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += vel_yaw_ratio * (cos(particles[i].theta) - cos(particles[i].theta + yaw_delta)) + dist_y(gen);
			particles[i].theta += yaw_delta + dist_theta(gen);
		}
		else {
			//add noise to x and y, theta is effectivley 0
			particles[i].x += vel_delta * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += vel_delta * sin(particles[i].theta) + dist_y(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {

		double min_dist = 10000.0;
		int tmp_id = 0;

		for (int j = 0; j < predicted.size(); j++) {

			double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (d < min_dist) {

				min_dist = d;
				tmp_id = j;
			}
		}

		observations[i].id = tmp_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//constants for weight calculation
	const double sigma_x2 = std_landmark[0] * std_landmark[0];
	const double sigma_y2 = std_landmark[1] * std_landmark[1];
	const double f = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	//sum of weights to normalize
	double sum_w = 0.0;

	unsigned int numObservations = observations.size();

	for (unsigned int i = 0; i < num_particles; i++) {
		//iterate through observations to find shortest distance between obsercation and particle
		for (unsigned int j = 0; j < numObservations; j++) {
			//transform from vehicle to map coords
			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = particles[i].x + (observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta));
			observation.y = particles[i].y + (observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta));
			
			//find closest landmark
			Map::single_landmark_s near_landmark;
			bool in_sensor_range = false;
			double short_dist = numeric_limits<double>::max();

			
			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
				// calculate Euclidean distance between transformed observation and the landmark
				double distance = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, observation.x, observation.y);

				if (distance < short_dist) {
					//update the value of short_dis
					short_dist = distance;

					//assign this landmark as the nearest one to the observation
					near_landmark = map_landmarks.landmark_list[k];

					//check if the distance is within sensor range
					if (distance < sensor_range) {
						in_sensor_range = true;
					}
				}
			}
			//if landmark is in range, calculate weight
			if (in_sensor_range == true) {
				//calculate weight
				double dx = observation.x - near_landmark.x_f;
				double dy = observation.y - near_landmark.y_f;

				//calculate multivariable-gaussian (weight)
				double weight = f * exp(-0.5 * ((dx * dx / sigma_x2) + (dy * dy / sigma_y2)));

				//final weight of the particle will be the product of each measurement's multivariable-gaussian probability density (weight)
				particles[i].weight *= weight;

				//sum of weights for normalization
				sum_w += particles[i].weight;
			}

		}
	}
	//weights normalization so that sum of weights = 1
	for (unsigned int l = 0; l < num_particles; ++l) {
		particles[l].weight /= sum_w;
		weights[l] = particles[l].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	static default_random_engine gen;

	discrete_distribution<> dis_particles(weights.begin(), weights.end());
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for (int i = 0; i < num_particles; i++) {
		new_particles[i] = particles[dis_particles(gen)];
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
