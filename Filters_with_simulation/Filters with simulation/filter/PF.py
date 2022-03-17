
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
rng = default_rng()

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        np.random.seed(2)
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma
        self.particles = init.particles
        self.particle_weight = init.particle_weight


        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for PF, remove pass                     #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.                                      #
        ###############################################################################
        
        # sample noise distribution
        M_k = self.M(u)
        noiseDim = len(M_k)
        mu_noise = np.zeros(noiseDim)
        noise_k = rng.multivariate_normal(mu_noise, M_k, size=self.n).T

        # evaluate particle motion
        self.particles = np.array([ self.gfun(self.particles[:, i], u + noise_k[:, i]) for i in range(0, self.n) ]).T
        self.particles[2,:] = wrap2Pi(self.particles[2,:])


        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


    def correction(self, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        # Get landmark coordinates
        m,n = self.particles.shape
        
        z1_1 = landmark1.getPosition()[0]
        z1_2 = landmark1.getPosition()[1]
        
        z2_1 = landmark2.getPosition()[0]
        z2_2 = landmark2.getPosition()[1]
        
        mean = np.array([0, 0, 0, 0])
        cov = np.eye(4)*3
        
        for i in range(n):
        	
        	measurement1 = self.hfun(z1_1, z1_2, self.particles[:,i])
        	particle_sub1 = measurement1 - z[0:2]
        	
        	measurement2 = self.hfun(z2_1, z2_2, self.particles[:,i])
        	particle_sub2 = measurement2 - z[3:5]
        	
        	particlesub = particle_sub1 - particle_sub2
        	
        	#calculate probability of or weight value
        	
        	zz = np.concatenate((particle_sub1,particle_sub2))
        	self.particle_weight[i] = multivariate_normal.pdf(zz,mean, cov)
        
        #Normalize the weight
        self.particle_weight = (self.particle_weight/(np.sum(self.particle_weight)))
        self.resample()

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.mean_variance()


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.zeros_like(self.particle_weight)
        W = np.cumsum(self.particle_weight)
        r = np.random.rand(1) / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
            new_weight[j] = 1 / self.n
        self.particles = new_samples
        self.particle_weight = new_weight
    

    def mean_variance(self):
        X = np.mean(self.particles, axis=1)
        sinSum = 0
        cosSum = 0
        for s in range(self.n):
            cosSum += np.cos(self.particles[2,s])
            sinSum += np.sin(self.particles[2,s])
        X[2] = np.arctan2(sinSum, cosSum)
        zero_mean = np.zeros_like(self.particles)
        for s in range(self.n):
            zero_mean[:,s] = self.particles[:,s] - X
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
        P = zero_mean @ zero_mean.T / self.n
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

