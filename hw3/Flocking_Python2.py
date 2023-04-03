###########################################################################
# March 2019, Orit Peleg, orit.peleg@colorado.edu
# Code for HW3 CSCI 4314/5314 Dynamic Models in Biology
###########################################################################
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio


class Flock:
    def __init__(self, n, attraction, repulsion, heading, randomness, velocity_max):
        self.N = n
        self.c1 = attraction  # Attraction Scaling factor
        self.c2 = repulsion  # Repulsion scaling factor
        self.c3 = heading  # Heading scaling factor
        self.c4 = randomness  # Randomness scaling factor
        self.vlimit = velocity_max  # Maximum velocity

        self.frames = 100  # No. of frames
        self.axl = 100  # Axis Limits
        self.limit = self.axl * 2
        self.pspread = 10  # Spread of initial positions (gaussian)
        self.vspread = 10  # Spread of initial velocitys (gaussian)
        self.timestepsize = 5

        # initiaized below
        self.p = None
        self.v = None
        self.fig = None
        self.ax = None
        self.line1 = None

    def initialize(self):
        # positions
        self.p = self.pspread*np.random.randn(2, self.N)
        # velocities
        self.v = self.vspread*np.random.randn(2, self.N)
        self.v = self.v / np.linalg.norm(self.v, axis=0)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def flocking(self):
        assert self.p is not None, "Call initialize() first"
        for i in range(0, self.frames):
            v1 = np.zeros((2, self.N))
            v2 = np.zeros((2, self.N))
            v3 = np.array([np.sum(self.v[0, :]) / self.N,
                           np.sum(self.v[1, :]) / self.N, ]) * self.c3
            if np.linalg.norm(v3) > self.vlimit:
                v3 *= self.vlimit / np.linalg.norm(v3)
            v4 = np.zeros((2, self.N))
            for n in range(0, self.N):
                for m in range(0, self.N):
                    if m != n:
                        r = self.p[:, m] - self.p[:, n]
                        if r[0] > self.limit/2:
                            r[0] = r[0]-self.limit
                        elif r[0] < -self.limit/2:
                            r[0] = r[0]+self.limit

                        if r[1] > self.limit/2:
                            r[1] = r[1]-self.limit
                        elif r[1] < -self.limit/2:
                            r[1] = r[1]+self.limit

                        # Compute distance between agents rmag
                        rmag = np.sqrt(r[0]**2 + r[1]**2)

                        # Compute attraction v1
                        v1[:, n] = v1[:, n] + self.c1 * r

                        # Compute Repulsion [non-linear scaling] v2
                        v2[:, n] = v2[:, n] - self.c2 * r / (rmag ** 2)

                # Compute random velocity component v4
                v4[:, n] = self.c4 * np.random.randn(2, 1).flatten()

                # Update velocity
                self.v[:, n] += v1[:, n] + v2[:, n] + v3 + v4[:, 0]
                if np.linalg.norm(self.v) > self.vlimit:
                    self.v = self.v / np.linalg.norm(self.v, axis=0)

            # Update position
            self.p += self.v * self.timestepsize

            # Periodic boundaries
            tmp_p = self.p

            tmp_p[0, self.p[0, :] > self.limit/2] = tmp_p[0,
                                                          self.p[0, :] > (self.limit/2)] - self.limit
            tmp_p[1, self.p[1, :] > self.limit/2] = tmp_p[1,
                                                          self.p[1, :] > (self.limit/2)] - self.limit
            tmp_p[0, self.p[0, :] < -self.limit/2] = tmp_p[0,
                                                           self.p[0, :] < (-self.limit/2)] + self.limit
            tmp_p[1, self.p[1, :] < -self.limit/2] = tmp_p[1,
                                                           self.p[1, :] < (-self.limit/2)] + self.limit
            self.p = tmp_p

            # with plt.ion():
            self.ax.cla()
            self.ax.scatter(self.p[0, :], self.p[1, :])
            # For drawing velocity arrows
            self.ax.quiver(self.p[0, :], self.p[1, :],
                           self.v[0, :], self.v[1, :])
            plt.xlim(-self.axl, self.axl)
            plt.ylim(-self.axl, self.axl)
            plt.pause(0.005)
            plt.savefig('bruh{}.png'.format(i))


n_agents = 100
c1 = 0.00001  # attraction
c2 = 0.01  # repulsion
c3 = 1.0   # heading
c4 = 0.01 / 5  # randomness
flock_py = Flock(n=n_agents, attraction=c1, repulsion=c2,
                 heading=c3, randomness=c4, velocity_max=1)
flock_py.initialize()
flock_py.flocking()
frames = []
for i in  range(100):
   image = imageio.v2.imread('bruh{}.png'.format(i))
   frames.append(image)
imageio.mimsave('c4div5.gif', frames)
