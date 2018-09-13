from collections import deque
import numpy as np
import argparse
import imutils
import cv2
from scipy.spatial.distance import cdist

COLOR_LOWER = (23, 54, 119)
COLOR_UPPER = (60, 192, 176)
NUM_PARTICLES = 500
POSITION_STD = 10
RADIUS_STD = 10

rng = np.random.RandomState()

# see 
# https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
def intersection_area(d, R, r):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.

    """
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta -
             0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
           )

areas = {}
def get_area(r):
    try:
        return areas[r]
    except KeyError:
        area = np.pi * (r ** 2)
        areas[r] = area
        return area

class ParticleFilter:

    def __init__(self, num_particles, x_bounds, y_bounds, r_bounds):
        """Create particle filter, each particle is (x,y,r).
        
        Args:
            num_particles (int): number of particles in the filter
        """
        
        # particles stored in n x 3 matrix, initialize randomly
        self.num_particles = num_particles
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.r_bounds = r_bounds
        self.initialize_uniformly()


    def initialize_uniformly(self):
        self.particles = np.hstack((rng.randint(self.x_bounds, 
                                                (self.num_particles,1)),
                                    rng.randint(self.y_bounds, 
                                                (self.num_particles,1)),
                                    rng.randint(self.r_bounds, 
                                                (self.num_particles,1))))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def elapse_time(self):
        """Step forward in time.  This is what enforces locality."""
        # using normal dist, guess as to good standard deviation
        updates = np.hstack((rng.normal(0, POSITION_STD, (self.num_particles,2)),
                             rng.normal(0, RADIUS_STD, (self.num_particles,1)))
                           )
        self.particles += np.asarray(updates.round(), dtype='int')
        self.particles = self.particles.clip([self.x_bounds[0], 
                                              self.y_bounds[0],
                                              self.r_bounds[0]],
                                             [self.x_bounds[1], 
                                              self.y_bounds[1],
                                              self.r_bounds[1]])
        
    def observe(self,mask):
        """Handle observations.
        
        Particles are re-weighted based on likelihood.
        """
        self.weights[:] = 0.001

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                                    
        enclosing_circles = [cv2.minEnclosingCircle(c) for c in contours]
        
        
        

        for i, particle in enumerate(self.particles):
            for ((x, y), radius) in enclosing_circles:
                distance = ((particle[0] - x) ** 2 + (particle[1] - y) ** 2) ** 0.5
                overlap = intersection_area(distance, radius, particle[2])
                self.weights[i] += 2 * overlap / (get_area(particle[2]) + get_area(radius))
        
        #matching_pixels = np.asarray(list(zip(*np.where(mask))))
        #if len(matching_pixels) > 0:
        #    distances = cdist(self.particles[:,:2], matching_pixels)
        #    for (x,y) in zip(*np.where(mask)):
        #        for i, particle in enumerate(self.particles):
        #            dists = distances[i,:]
        #            self.weights[i] += len(np.where(dists < particle[2])[0])
        
            
        # normalize
        if np.sum(self.weights) > 0:
            self.weights[:] = self.weights / np.sum(self.weights)
        else:
            print("********** REINITIALIZING UNIFORMLY ******************")
            self.initialize_uniformly()

    def draw_most_likely(self, frame):
        # single most likely
        (x,y,radius) = self.particles[np.argmax(self.weights)]
        
        # weighted average
        #(x,y,radius) = np.sum(self.particles * 
        #                      np.broadcast_to(self.weights, 
        #                                      (3,self.num_particles)
        #                                     ).transpose(), 0)

        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                
    def resample(self):
        new_particle_indices = rng.choice(np.arange(self.num_particles),
                                          self.num_particles,
                                          p = self.weights)
        new_particles = self.particles[new_particle_indices]
        self.particles = new_particles
        
    def draw_all(self, frame):
        for (x,y,radius) in self.particles:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)


    def return_most_likely(self, frame):
        # weighted average
        (x,y,radius) = np.sum(self.particles * 
                              np.broadcast_to(self.weights, 
                                              (3,self.num_particles)
                                             ).transpose(), 0)
	return (x,y,radius)
               
            
