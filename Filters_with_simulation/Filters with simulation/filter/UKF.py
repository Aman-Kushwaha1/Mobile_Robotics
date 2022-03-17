
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)



    def prediction(self, u):
        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ##############################################################################
        kappa = self.kappa_g
        Mcov = self.M(u)
        sigma_noise = block_diag(Mcov)
        noiseDim = len(sigma_noise)
        
        mu_noise = np.zeros((noiseDim, 1))

        # predicted mean and sigma
        meanb = np.vstack((np.array([X], dtype="float").T, mu_noise))
        sigma_block = block_diag(P, sigma_noise)
        
        #cholesky decomposition for generating mean and sigma points
        n = len(sigma_block)
        L = np.linalg.cholesky(sigma_block)
        Lnew = np.sqrt(n+self.kappa_g)*L


        num = 2*n + 1
        s_p = np.full((n, num), meanb)
        
        s_p[:, 1:n+1]+= Lnew
        
        s_p[:, n+1:2*n+1]-= Lnew

        
        w = np.zeros(num)
        w[0] = kappa/(n+ kappa) 
        w[1:num] = .5/(n + kappa)

        
        
        s_pnew = np.array([self.gfun(s_p[:, i], u + s_p[3:, i]) for i in range(0, num)]).T
        
        s_pnew = np.vstack((s_pnew, s_p[3:]))

        # new mean
        meannew = np.sum(w * s_pnew, axis=1, keepdims=True)

        # new sigma estimate
        
        
        sigmanew = np.zeros_like(sigma_block)
        
        deltaFSigmaPoints = s_pnew -  meannew 
        
        
        for i in range(num):
            deltaFSigmaPoint = deltaFSigmaPoints[:, i]
            sigmanew+= w[i] * np.outer(deltaFSigmaPoint, deltaFSigmaPoint) 

        
        #Final update
        P_pred = sigmanew[0:len(X), 0:len(X)]
        
        X_pred = meannew.T[0,0:len(X)].astype("float")

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):

        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        mu_k = np.array([X_predict], dtype="float").T
        sigma_k = P_predict

        n = len(mu_k)
        N = 2*n + 1 

        # compute cholesky decomposition
        L = np.linalg.cholesky(sigma_k)
        LPrime = np.sqrt(n+self.kappa_g)*L

        # Compute sigma points
        sigmaPoints = np.full((n,N), mu_k)
        sigmaPoints[:, 1:n+1]+= LPrime
        sigmaPoints[:, n+1:2*n+1]-= LPrime

        # Get landmark coordinates
        m_x1 = landmark1.getPosition()[0]
        m_y1 = landmark1.getPosition()[1]
        m_x2 = landmark2.getPosition()[0]
        m_y2 = landmark2.getPosition()[1]
        
        
        
        z_1 = np.array([z[0:2]]).T
        z_2 = np.array([z[3:5]]).T
        z_k = np.vstack((z_1, z_2))               


        # Finding weights
        w = np.zeros(N)
        w[0] = self.kappa_g/(n+self.kappa_g) 
        w[1:N] = .5/(n + self.kappa_g)

        hsp1 = np.array([self.hfun(m_x1, m_y1, sigmaPoints[:, i]) for i in range(0, N)]).T
        
        hsp2 = np.array([self.hfun(m_x2, m_y2, sigmaPoints[:, i]) for i in range(0, N)]).T
        
        hsp = np.vstack((hsp1, hsp2))

        # Compute Inovation 
        z_kBar = np.sum(w * hsp, axis=1, keepdims=True)
        
        nu_k = z_k - z_kBar

        # iNNOVATION 
        Q_k = block_diag(self.Q, self.Q)
        S_k = Q_k
        Z_k = 0
        hsp = hsp -  z_kBar 
        
        for i in range(0, N):
            deltaHSigmaPoint = hsp[:, i]
            S_k+= w[i] * np.outer(deltaHSigmaPoint, deltaHSigmaPoint)

            deltaSigmaPoint = sigmaPoints[:, i] - mu_k.T           
            Z_k+= w[i] * np.outer(deltaSigmaPoint, deltaHSigmaPoint)

        
        K_k = np.matmul(Z_k, np.linalg.inv(S_k))

        
        mu_k = mu_k + np.matmul(K_k, nu_k)
        
        sigma_k = sigma_k - np.matmul(np.matmul(K_k, S_k), K_k.T)      

       
        X = mu_k.T[0].astype("float")
        
        X[2] = wrap2Pi(X[2])
        
        P = sigma_k

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
