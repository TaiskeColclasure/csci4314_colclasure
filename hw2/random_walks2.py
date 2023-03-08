import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pylab


class Random_Walks():
    def random_walks(self):
        N = 500  # no of steps per trajectory
        realizations = 50  # number of trajectories
        v = 1.0  # velocity (step size)

        # the width of the random walk turning angle distribution
        # (the lower it is, the more straight the trajectory will be)
        theta_s_array = [round(math.pi / 24, 4), round(math.pi / 12, 4), round(math.pi / 3, 4)]

        # weighting given to the directional bias (hence (1-w) is the weighting given to correlated motion)
        w_array = np.linspace(.04, .08, num=100)

        # initial ratio between bias and correlated motion
        ratio_theta_s_brw_crw = 1

        # whether to plot realizations of random walks
        plot_walks = False 
        count = 0
        tBucket = np.matrix(list(range(1, 501)))
        for x in range(len(w_array) -1):
            tBucket = np.vstack((tBucket, np.matrix(list(range(1, 501)))))

        # TODO: calc navigational efficiency
        navEffMatrix = []
        for w_i in range(len(w_array)):
            w = w_array[w_i]
            theta_s_crw = round(math.pi / 30, 4)#np.multiply(ratio_theta_s_brw_crw, theta_s_array[theta_s_i])
            theta_s_brw = round(math.pi / 3, 4)#theta_s_array[theta_s_i]
            x, y = self.BCRW(N, realizations, v, theta_s_crw, theta_s_brw, w)
            navBucket = []
            for i in range(500):
                temp = 0
                for j in range(50):
                    temp += x[j][i] / (i + 1)
                navBucket.append(temp / 50)
            navEffMatrix.append(navBucket)
            print(w_i)
        plt.show()
        # TODO: plot navigational efficiency
        navEffMatrix = np.matrix(navEffMatrix)
        #plt.figure('brus')
        #plt.title('Navigational Efficiency of Random Walks')
        #plt.plot(tBucket.T, navEffMatrix.T, label=w_array)
        #plt.ylim(.95, .975)
        #plt.legend()
        #plt.show()
        maxBucket = (0,0)
        arrayForm = navEffMatrix.tolist()
        for i in range(navEffMatrix.shape[0]):
            if arrayForm[i][-1] > maxBucket[0]:
                maxBucket = (arrayForm[i][-1], i)
        print(maxBucket, w_array[maxBucket[1]])


    # The function generates 2D-biased correlated random walks
    def BCRW(self, N, realizations, v, theta_s_crw, theta_s_brw, w):
        X = np.zeros([realizations, N])
        Y = np.zeros([realizations, N])
        theta = np.zeros([realizations, N])
        X[:, 0] = 0
        Y[:, 0] = 0
        theta[:, 0] = 0  # theta_0 direction (positive x)

        for realization_i in range(realizations):
            for step_i in range(1, N):
                theta_crw = theta[realization_i][step_i - 1] + (theta_s_crw * 2.0 * (np.random.rand(1, 1) - 0.5))
                theta_brw = (theta_s_brw * 2.0 * (np.random.rand(1, 1) - 0.5))

                X[realization_i, step_i] = X[realization_i][step_i - 1] + (v * (w * math.cos(theta_brw))) + (
                            (1 - w) * math.cos(theta_crw))
                Y[realization_i, step_i] = Y[realization_i][step_i - 1] + (v * (w * math.sin(theta_brw))) + (
                            (1 - w) * math.sin(theta_crw))

                current_x_disp = X[realization_i][step_i] - X[realization_i][step_i - 1]
                current_y_disp = Y[realization_i][step_i] - Y[realization_i][step_i - 1]
                current_direction = math.atan2(current_y_disp, current_x_disp)

                theta[realization_i, step_i] = current_direction

        return X, Y


rdm_plt = Random_Walks()
rdm_plt.random_walks()
