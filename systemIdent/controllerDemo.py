import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

# env = gym.make('Reacher-v4', render_mode='human')
env = gym.make('Reacher-v4', render_mode = 'human')

observation, info = env.reset() 

#d_cylinder_base = 0.01
#link0 length = 0.1
#link1 length = 0.1
#end effector d = 0.01

#total arm lenght 0.21

############ DECLARE PARAMETERS ############
m1 = 2.809173
m2 = 0.41335134

r1 = 0.01/2
r2 = 0.00001


l1 = 0.1
l2 = 0.1

d1 = l1/2
d2 = l2/2

I_1 = 1 / 12 * m1 * (3 * r1**2 + l1**2)
I_2 = 1 / 12 * m2 * (3 * r2**2 + l2**2)

def M(obs):

    c1 = obs[0]
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]


    alpha = (
        m1 * d1**2 + I_1
        + m2 * (l1**2 + d2**2)
        + I_2
    )

    beta = m2 * l1 * d2
    gamma = m2 * d2**2 + I_2
    M_11 = alpha + 2 * beta * c2
    M_12 = gamma + beta * c2
    M_21 = gamma + beta * c2
    M_22 = gamma

    return np.array([[M_11, M_12], [M_21, M_22]])


def C(obs):
    c1 = obs[0]
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]

    d1 = l1 / 2
    d2 = l2 / 2
    alpha = (
        m1 * d1**2
        + I_1
        + m2 * (l1**2 + d2**2)
        + I_2
    )
    beta = m2 * l1 * d2
    gamma = m2 * d2**2 + I_2

    q1_p = obs[6]
    q2_p = obs[7]

    C_11 = -beta * s2 * q2_p
    C_12 = -beta * s2 * (q1_p + q2_p)
    C_21 = beta * s2 * q1_p
    C_22 = 0

    return np.array([[C_11, C_12], [C_21, C_22]])


c1 = observation[0]
c2 = observation[1]
s1 = observation[2]
s2 = observation[3]

xTarget = observation[4]
yTarget = observation[5]

q1_p = observation[6]
q2_p = observation[7]
q1_p_prev = q1_p
q2_p_prev = q2_p

xFingertip = observation[8] + xTarget
yFingertip = observation[9] + yTarget


# SET GOAL ANGLES IN RADIANS
q_r = np.array([[0.5], [2.3]])
q_r_p = np.array([[0.0], [0.0]])
q_r_pp = np.array([[0.0], [0.0]])

q1 = np.angle(c1 + 1j*s1)
q2 = np.angle(c2 + 1j*s2)

q = np.array([[q1], [q2]])
q_p = np.array([[q1_p], [q2_p]])

Kp = np.array([[10.2, 0], [0, 10.2]])
Kd = np.array([[7.2, 0], [0, 7.2]])
Kp = np.array([[0, 0], [0, 0]])
Kd = np.array([[0, 0], [0, 0]])

q_r_pp = np.array([[1250], [5750]])

for n in range(1, 1000):


    v = q_r_pp + Kd @ (q_r_p - q_p) + Kp @ (q_r - q)
    q_r_pp = np.array([[0.0], [0.0]])

    tau = M(observation) @ v + C(observation) @ q_p
    action = (*tau[0], *tau[1])
    # print(action)

    c1 = observation[0]
    c2 = observation[1]
    s1 = observation[2]
    s2 = observation[3]

    xTarget = observation[4]
    yTarget = observation[5]

    q1_p = observation[6]
    q2_p = observation[7]

    q1 = np.angle(c1 + 1j*s1)
    q2 = np.angle(c2 + 1j*s2)

    q = np.array([[q1], [q2]])
    q_p = np.array([[q1_p], [q2_p]])

    print("expected angles: ", q_r)
    print("angles: ",q)

    observation, reward, terminated, truncated, info = env.step(action)

env.close()
