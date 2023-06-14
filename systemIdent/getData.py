import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math


# env = gym.make('Reacher-v4', render_mode='human')
env = gym.make('Reacher-v4')


observation, info = env.reset() 

#d_cylinder_base = 0.01
#link0 length = 0.1
#link1 length = 0.1
#end effector d = 0.01

#total arm lenght 0.21

r1 = 0.01/2
r2 = 0.00001

l1 = 0.1
l2 = 0.1

d1 = l1/2
d2 = l2/2

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


tp = 0.05
f1 = 0.07
f2 = 0.07
t = np.arange(0, 250, tp)


tau1arr = []
tau2arr = []


A = np.zeros((1, 2))
Tau = np.zeros((1,1))

counter = 0
howmany = 0
for n in range(1, len(t)):
    # print(np.sin(2*np.pi*f*t))

    # 0.017 -> 1 degree
    if (counter == 500):
        counter = 0
        f1+=0.05
        f2+=0.05
        howmany+=1

    tau1 = 0.02 * np.sin(2 * np.pi * f1 * t[n])
    tau2 = 0.02 * np.sin(2 * np.pi * f2 * t[n] + np.pi/3)
    tau1arr.append(tau1)
    tau2arr.append(tau2)

    action = (tau1, tau2)

    observation, reward, terminated, truncated, info = env.step(action)

    c1 = observation[0]
    c2 = observation[1]
    s1 = observation[2]
    s2 = observation[3]

    xTarget = observation[4]
    yTarget = observation[5]

    q1_p = observation[6]
    q2_p = observation[7]

    #numerical derivative
    q1_pp = (q1_p - q1_p_prev)/(t[n] - t[n-1])
    q2_pp = (q2_p - q2_p_prev)/(t[n] - t[n-1])

    xFingertip = observation[8] + xTarget
    yFingertip = observation[9] + yTarget

    A11 = (d1**2 + 1/12 * (3*r1**2 + l1**2)) * q1_pp
    A12 = ((l1**2 + d2**2) + 1/12 *(3*r2**2 + l2**2) + 2*l1*d2*c2)* q1_pp \
            + (d2**2 + 1/12 *(3*r2**2 + l2**2) + l1*d2*c2)*q2_pp \
            - l1*d2*q2_p*q1_p - l1*d2*s2*(q1_p + q2_p)* q2_p

    A21 = 0
    A22 = (d2**2 + l1*d1*c1)*q1_pp + 1/12 * (3*r2**2 + l2**2)*(q1_pp+q2_pp) + d2**2 * q2_pp + l1*d2*s2*q1_p
    
    A = np.append(A, [[A11, A12]], axis=0)
    A = np.append(A, [[A21, A22]], axis=0)
    Tau = np.append(Tau, [[tau1], [tau2]], axis=0)


    q1_p_prev = q1_p
    q2_p_prev = q2_p

    counter += 1


    # if terminated or truncated:
        # observation, info = env.reset()

env.close()


print("HOWMANY: ", f1)
#delete init row
A = np.delete(A, 0, 0)
Tau = np.delete(Tau, 0, 0)

print("A: \n", A)
print("Tau: \n", Tau)

print("---\nA shape: ", A.shape)
print("Tau shape: ", Tau.shape)

M = np.linalg.pinv(A) @ Tau

print("M: \n", M)


plt.plot(tau1arr)
plt.plot(tau2arr)
plt.show()