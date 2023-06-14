import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
from poly3 import Poly3

# env = gym.make('Reacher-v4', render_mode='human')
env = gym.make("Reacher-v4", render_mode="human")

observation, info = env.reset()

# d_cylinder_base = 0.01
# link0 length = 0.1
# link1 length = 0.1
# end effector d = 0.01

# total arm lenght 0.21
end = 2000 / 0.01

# traj_gen = Poly3(np.array([0.0, 0.0]), np.array([math.pi / 4, math.pi / 6]), end)


############ DECLARE PARAMETERS ############
m1 = 2.809173
m2 = 0.41335134

r1 = 0.01 / 2
r2 = 0.00001

l1 = 0.1
l2 = 0.1 + 0.01

d1 = l1 / 2
d2 = l2 / 2

I_1 = 1 / 12 * m1 * (3 * r1**2 + l1**2)
I_2 = 1 / 12 * m2 * (3 * r2**2 + l2**2)


def setNewDestination(observation):
    global xTarget, yTarget, xFingertip, yFingertip, traj_gen
    xTarget = observation[4]
    yTarget = observation[5]

    xFingertip = observation[8] + xTarget
    yFingertip = observation[9] + yTarget

    ah1, ah2, ath = caluclateInvertKinematics(xTarget, yTarget)
    ad1, ad2, atd = caluclateInvertKinematics(xFingertip, yFingertip)

    if yTarget < 0:
        qh = [ath + ah1, ah2 - np.pi]
        qd = [atd + ad1, ad2 - np.pi]
    else:
        qh = [ath - ah1, np.pi - ah2]
        qd = [atd - ad1, np.pi - ad2]

    traj_gen = Poly3(np.array([qh[0], qh[1]]), np.array([qd[0], qd[1]]), end)


def caluclateInvertKinematics(x, y, l1=0.1, l2=0.11):
    # q_r -> destinated angles ... calculate q_r using invert kinematicsx
    dist = math.sqrt(x**2 + y**2)
    cos1 = ((dist**2) + (l1**2) - (l2**2)) / (2 * dist * l1)
    angle1 = np.arccos(cos1)
    cos2 = ((l2**2) + (l1**2) - (dist**2)) / (2 * l2 * l1)
    angle2 = np.arccos(cos2)
    atan = np.arctan2(y, x)
    return angle1, angle2, atan


def M(obs):
    c1 = obs[0]
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]

    alpha = m1 * d1**2 + I_1 + m2 * (l1**2 + d2**2) + I_2
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
    alpha = m1 * d1**2 + I_1 + m2 * (l1**2 + d2**2) + I_2
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


q1_p = observation[6]
q2_p = observation[7]
q1_p_prev = q1_p
q2_p_prev = q2_p


# SET GOAL ANGLES IN RADIANS
q_r = np.array([[-2.5], [2.3]])
q_r_p = 0
q_r_pp = 0

q1 = np.angle(c1 + 1j * s1)
q2 = np.angle(c2 + 1j * s2)

q = np.array([[q1], [q2]])
q_p = np.array([[q1_p], [q2_p]])

Kp = np.array([[100, 0], [0, 100]])
Kd = np.array([[18, 0], [0, 18]])


switchedGoal = True

setNewDestination(observation)
i = 0
for n in range(1, 500000):
    c1 = observation[0]
    c2 = observation[1]
    s1 = observation[2]
    s2 = observation[3]

    xTarget = observation[4]
    yTarget = observation[5]

    q1 = np.angle(c1 + 1j * s1)
    q2 = np.angle(c2 + 1j * s2)

    q = np.array([q1, q2])
    q_p = np.array([observation[6], observation[7]])
    xFingertip = observation[8] + xTarget
    yFingertip = observation[9] + yTarget

    q_r, q_r_p, q_r_pp = traj_gen.generate(i)

    v = Kd @ (q_r_p - q_p) + Kp @ (q_r - q)

    tau = M(observation) @ v + C(observation) @ q_p

    action = (tau[0], tau[1])

    observation, reward, terminated, truncated, info = env.step(action)
    i += 0.01

    if (abs(xFingertip - xTarget) <= 0.01) and (abs(yFingertip - yTarget) <= 0.01):
        observation, info = env.reset()
        setNewDestination(observation)

env.close()
