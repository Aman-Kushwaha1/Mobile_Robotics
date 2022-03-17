import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        
        # Estimate mean
        mu_new = self.gfun(X, u)
        
        F = self.Gfun(X, u)
        W = self.Vfun(X, u)

        # sigma update
        mm = np.matmul(F, P)
        sig = np.matmul(mm , F.T) + np.matmul(np.matmul(W, self.M(u)), W.T)  

        P_pred = sig
        X_pred = mu_new

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
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
	 
        mu_kBar = np.array([X_predict]).T

        #z calculation
        mx1 = landmark1.getPosition()[0]
        my1 = landmark1.getPosition()[1]
        mx2 = landmark2.getPosition()[0]
        my2 = landmark2.getPosition()[1]
        sen1 = np.array([z[0:2]]).T
        sen2 = np.array([z[3:5]]).T
        z_1 = sen1
        z_2 = sen2
        z_k = np.vstack((z_1, z_2))

        # stacked version of zk
        k1 = np.array([self.hfun(mx1, my1, mu_kBar)]).T
        k2 = np.array([self.hfun(mx2, my2, mu_kBar)]).T
        hk = np.vstack((k1, k2))

        measure = z_k - hk
        measure[0] = wrap2Pi(measure[0])
        measure[2] = wrap2Pi(measure[2])

        # Evaluate Jocabians
        H_k1 = self.Hfun(mx1, my1, mu_kBar, z_1)
        H_k2 = self.Hfun(mx1, my1, mu_kBar, z_2)
        Hstack = np.vstack((H_k1, H_k2))



        # finding covariance
        Qstack = block_diag(self.Q, self.Q)
        S_k = np.matmul(np.matmul(Hstack, P_predict), Hstack.T) + Qstack

        
        Kalmangain = np.matmul(np.matmul(P_predict, Hstack.T), np.linalg.inv(S_k.astype('float')))

        # mean correction
        mu_pred = mu_kBar + np.matmul(Kalmangain, measure)

        # sigma correction
        mm1 = np.eye(3) - np.matmul(Kalmangain, Hstack)
        mm = np.matmul(mm1, P_predict)
        sigmaupdate = np.matmul(mm, mm1.T) + np.matmul(np.matmul(Kalmangain, Qstack), Kalmangain.T)

        # final update
        P = sigmaupdate
        X = mu_pred.T[0].astype('float')

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
