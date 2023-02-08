import matplotlib.pyplot as plt
import math
import numpy as np
import random
import os
import imageio

#global
filenames = []
time = []
radiusContainer = []
#################################################################
def Polymer_MD_Python():
    # initialize parameters
    N = 25 #no. of particles
    T = 0.05 #temperature
    dt = 0.0005 #integration time step
    steps = 200000 #time steps

    epsilon_LJ = 0
    cutff_LJ = 2.5
    spring_coeff = 5
    min_sep = 1.122

    L = min_sep*N
    print_interval = 1000
####################################################################
    #initialize x coordinates
    x = initial_configuration(min_sep, N)
    pairs = []
#####################################################################
    #main loop
    frame = 1
    for step_i in range(0, steps): #molecular dynamics loop
        x, pairs = steepest_descent(N, x, dt, cutff_LJ, epsilon_LJ, min_sep, spring_coeff, T)
        radiusContainer.append(calc_radiusGyration(N, x))
        time.append(step_i)
        if (np.mod(step_i-1,print_interval) == 0):  #print every 1000 steps
            mytitle = ["step=",str(step_i), "N=", str(N), "L=", str(L)]
            print(mytitle)
            visualize_particles(N, x, L, pairs, mytitle, frame)
            frame +=1



# Initial coordinates of every amino acid.
def initial_configuration(initial_min_sep, N):
    x = np.zeros((N, 2))
    for i in range(0,N):
        x[i][0] = initial_min_sep*i - (initial_min_sep*N/2)
    return x

def steepest_descent(N,x,dt, cutoff_LJ,epsilon_LJ,min_sep,spring_coeff,T):
    F_particles,_,pairs = forces(N,x,cutoff_LJ,epsilon_LJ,min_sep,spring_coeff)
    F = F_particles
    x = x + (dt * F) + np.dot(T, (np.random.rand(x.shape[0], x.shape[1]) - 0.5))
    return x, pairs


def all_interactions(N,x,cutoff): #obtain interacting pairs
    ip =0
    connector = []
    pair = []
    for i in range(0, N-1):
        for j in range(i+1, N):
            distance = x[j,:]-x[i,:]  # distance : (1x2)
            if np.linalg.norm(distance) < cutoff:
                ip = ip + 1
                pair.append([i,j])
                connector.append([distance])
    return ip, pair, connector

#Obtain interacting pairs
def spring_interactions(N,x):
    ip = 0
    connector = []
    pair = []
    for i in range(0, N-1):
        j = i+1
        distance = x[j,:]-x[i,:]
        ip += 1
        pair.append([i,j])
        connector.append([distance])
    return ip, pair, connector

def forces(N,x,cutoff_LJ,epsilon_LJ,min_sep,spring_coeff):
    F = np.zeros((N,2))
    P = np.zeros((N,2))
    # LJ Forces
    no, pair, connector = all_interactions(N,x,cutoff_LJ) #interacting pairs
    for i in range(0, no):
        FORCE = force_LJ(connector[i], epsilon_LJ)
        F[pair[i][0]] = F[pair[i][0]]-FORCE
        F[pair[i][1]]=F[pair[i][1]]+FORCE #action = reaction
        P[pair[i][0]]=P[pair[i] [0]]+(np.sum(FORCE* connector[i], axis=0))
        P[pair[i][1]]=P[pair[i][1]]+(np.sum(FORCE* connector[i], axis=0))

    #Spring Forces
    no, pair, connector = spring_interactions(N, x) #interacting pairs
    for i in range(0,no):
        FORCE = force_springs(connector[i], spring_coeff, min_sep)
        F[pair[i][0]]=F[pair[i][0]]-FORCE
        F[pair[i][1]]=F[pair[i][1]]+FORCE # action = reaction;
        P[pair[i][0]]=P[pair[i][0]]+(np.sum(FORCE* connector[i], axis=0))
        P[pair[i][1]]=P[pair[i][1]]+(np.sum(FORCE* connector[i], axis=0))
    return F, P, pair

def force_springs(r_vector,spring_coeff_array,min_sep):
    r2 = np.sum(np.square(r_vector), axis = 1)
    r = np.sqrt(r2)
    curr_force = np.zeros((len(r2),2))
    val_1 = np.multiply(np.subtract(r,min_sep), (np.divide(r_vector[0][0], r)), out=None)
    val_2 = np.multiply(np.subtract(r,min_sep), (np.divide(r_vector[0][1], r)), out=None)
    curr_force[0][0] = np.multiply(np.transpose(-spring_coeff_array), val_1 )
    curr_force[0][1] = np.multiply(np.transpose(-spring_coeff_array), val_2)
    return curr_force

def force_LJ(r_vector, epsilon_LJ):
    r = np.linalg.norm(r_vector)
    force_LJ = 24*epsilon_LJ*np.dot((np.dot(2,r**(-14))-r**(-8)),r_vector)
    return force_LJ

colorBucket = [
        '#3791cd',
        '#4099c8',
        '#48a0c3',
        '#51a8be',
        '#5aafba',
        '#63b7b5',
        '#6bbeb0',
        '#74c6ab',
        '#7dcea6',
        '#86d5a1',
        '#8edd9d',
        '#97e498',
        '#a0ec93',
        '#a6eb8b',
        '#acea84',
        '#b2ea7c',
        '#b8e974',
        '#bee86d',
        '#c4e765',
        '#d0e656',
        '#d6e54e',
        '#dce446',
        '#e2e43f',
        '#e8e337',
        '#f0eb37'
        ]
def visualize_particles(N, x, L, pairs, mytitle, frame):
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.subplot(2,1,1,aspect=2/2)
    X = [i[0] for i in x]
    Y = [i[1] for i in x]
    colors = (0,0,0)
    plt.ylim(top=10,bottom=-10)
    plt.xlim([-10,10])
    plt.scatter(X, Y, c=colorBucket, s=5)
    plt.title('Shape at t={}'.format(mytitle[1]))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2,1,2)
    plt.plot(time, radiusContainer)
    plt.title('Radius of Gyration')
    plt.xlabel('Time')
    plt.ylabel('R(t)')


    
    filename = 'f{}.png'.format(frame)
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)
    frame += 1
    return
def calc_radiusGyration(N, x):
    sumi = 0
    for i in range(N-1):
        sumj = 0
        for j in range(N-1):
            if j != i:
                distance = math.dist(x[i],x[j])
                sumj += distance ** 2
        sumi += sumj
    result = sumi / (2 * (N ** 2))
    return result

Polymer_MD_Python()
# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        print(filename)
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
