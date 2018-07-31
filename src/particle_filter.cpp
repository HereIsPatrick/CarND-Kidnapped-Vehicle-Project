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
#include <random>

#include "particle_filter.h"

#define EPS 0.001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (is_initialized) {
        return;
    }

    // Number of particles
    num_particles = 100;

    // Get standard deviations
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // Set Normal distribution
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    default_random_engine gen;

    // Generate particles
    for (int i = 0; i < num_particles; i++) {

        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
    }

    // Initialized.
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // Get standard deviations
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // Set normal distributions
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);
    default_random_engine gen;

    // Calculate new state
    for (int i = 0; i < num_particles; i++) {

        double theta = particles[i].theta;

        if ( fabs(yaw_rate) < EPS ) { // Drive stright, when yaw not change
            particles[i].x += velocity * delta_t * cos( theta );
            particles[i].y += velocity * delta_t * sin( theta );
        } else {
            particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
            particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
            particles[i].theta += yaw_rate * delta_t;
        }

        // Noise added.
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // Step. for each observation assign the nearest neighbor particular landmark.
    for (int i = 0; i < observations.size(); i++) {

        // Initialize minimum distance
        double minDistance = numeric_limits<double>::max();

        // Initialize map id with zero.
        int mapId = 0;

        // Step. for this observation, calculate minimum distance of predicted.
        for (int j = 0; j < predicted.size(); j++ ) {

            double xDistance = observations[i].x - predicted[j].x;
            double yDistance = observations[i].y - predicted[j].y;

            double distance = sqrt(xDistance * xDistance + yDistance * yDistance);

            // Step. find the minimum distance(between observation and predicted)
            if ( distance < minDistance ) {
                minDistance = distance;
                mapId = predicted[j].id;
            }
        }

        // Step. assign map id to observation of minimum distance
        observations[i].id = mapId;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
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

    for (int i = 0; i < num_particles; i++) {

        // Get particle
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // Hold the landmark with in sensor range.
        vector<LandmarkObs> predictions;

        // For each land mark.
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

            // Get landmark
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;

            // Save the landmark into predictions within sensor range.
            if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

                // add prediction to vector
                predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

        // Transform observations from vehicle coordinates to map coordinates
        vector<LandmarkObs> transformed_os;
        for (int j = 0; j < observations.size(); j++) {
            double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
            double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
            transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
        }

        // Associate observations with landmark.
        dataAssociation(predictions, transformed_os);

        // Initialize particle weight
        particles[i].weight = 1.0;

        for (int j = 0; j < transformed_os.size(); j++) {

            //
            double obs_x, obs_y, pred_x, pred_y;
            obs_x = transformed_os[j].x;
            obs_y = transformed_os[j].y;

            int associated_prediction = transformed_os[j].id;

            // Get x,y  of the prediction
            for (int k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == associated_prediction) {
                    pred_x = predictions[k].x;
                    pred_y = predictions[k].y;
                }
            }

            // Calculate weight for this observation with multivariate Gaussian
            double std_lm_x = std_landmark[0];
            double std_lm_y = std_landmark[1];
            double obs_w = ( 1/(2*M_PI*std_lm_x*std_lm_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(std_lm_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_lm_y, 2))) ) );

            // Product of this observation weight with total observations weight
            particles[i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Weights for particle
    vector<double> weights;

    // Initialize max weight.
    double maxWeight = numeric_limits<double>::min();

    for(int i = 0; i < num_particles; i++) {

        // Fill weights from particles.
        weights.push_back(particles[i].weight);

        // Get the max weight.
        if ( particles[i].weight > maxWeight ) {
            maxWeight = particles[i].weight;
        }
    }

    // Create distributions.
    uniform_real_distribution<double> random_weight(0.0, maxWeight);
    uniform_int_distribution<int> particle_index(0, num_particles - 1);
    default_random_engine gen;

    // Create particle index.
    int index = particle_index(gen);

    double beta = 0.0;

    // Resample by Wheel algorithm.
    vector<Particle> resampleParticles;
    for(int i = 0; i < num_particles; i++) {
        beta += random_weight(gen) * 2.0;
        while( beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }

        // Select the particle
        resampleParticles.push_back(particles[index]);
    }

    particles = resampleParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
