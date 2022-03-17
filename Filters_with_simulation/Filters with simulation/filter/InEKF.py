
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        adjX = np.array([ [self.mu[0,0], self.mu[0,1], self.mu[1,2]], [self.mu[1,0], self.mu[1,1], -1*self.mu[0,2]], [0, 0, 1]])
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.propagation(u_se2, adjX)

    def propagation(self, u, adjX):
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################
        self.X_pred = np.matmul(expm(self.mu), u)
        
        
        A = np.identity(3)
        #m1 = np.matmul(A,self.mu)
        #m2 = np.matmul(self.mu, A.T)
        
        mm = np.matmul(adjX,self.W)
        self.P_pred = np.matmul(mm, adjX.T)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
    def correction(self, Y1, Y2, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        m11 = landmark1.getPosition()[0]
        m12 = landmark1.getPosition()[1]
        m21 = landmark2.getPosition()[0]
        m22 = landmark2.getPosition()[1]
        H = np.array([ [m12, -1, 0], [-1*m11, 0, -1], [m22, -1, 0], [-m21, 0, -1] ] )
        
        mm = np.matmul(H,self.P_pred)
        
        Z11 = np.zeros((2,2))
        noise = np.asarray(np.bmat([[self.V, Z11], [Z11, self.V]]))
        
        S = np.matmul(mm,H.T) + noise
        
        mm = np.matmul(self.P_pred, H.T)
        L = np.matmul(mm, np.linalg.inv(S) )
        
        #Update steps
        mm1 = np.identity(3) - np.matmul(L,H)
        mm = np.matmul(mm1,self.P_pred)
        cc = np.matmul(mm, mm1.T)
        
        mmm2 = np.matmul(L,noise)
        mmm2 = np.matmul(mmm2, L.T)
        
        self.Sigma = cc + mmm2                                 #corrected covariance
        
        Y = np.concatenate((Y1, Y2))
        Z11 = np.zeros((3,3))
        x_new = np.asarray(np.bmat([[self.P_pred, Z11], [Z11, self.P_pred]]))
        b = np.concatenate((landmark1.getPosition(), landmark2.getPosition()))
        
        mm = np.matmul(x_new, Y)
        
        nn = np.delete(mm, 2, 0)
        nn = np.delete(nn, 4, 0)
        
        mm1 = np.matmul(L, nn-b)
        
        #Converting it into wedge notation
        w1 = mm1[0]
        w2 = mm1[1]
        w3 = mm1[2]
        
        wedge = np.array([ [0, w3, -w2], [-w3, 0, w1], [w2, -w1, 0] ])
        
        x = np.matmul(expm(wedge), self.P_pred)
        
        
        mq = Rot.from_matrix(x)
        X = mq.as_euler('xyz')
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
