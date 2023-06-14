import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
from sysidentpy.utils.generate_data import get_miso_data


env = gym.make('Reacher-v4', render_mode='human')

observation, info = env.reset() 



#d_cylinder_base = 0.01
#link0 length = 0.1
#link1 length = 0.1
#end effector d = 0.01



tp = 0.01
f = 0.5
t = np.arange(0, 100, tp)

observation, info = env.reset()
angleJoint0 = math.acos(observation[0])
angleJoint1 = math.acos(observation[1])


joint0goal = False

for _ in range(100000):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    # print(np.sin(2*np.pi*f*t))

    # 0.017 -> 1 degree
    if (not((angleJoint0 <= 0.017) and (angleJoint0 >= 0.0))):
        action = (-0.0001, 0.00)
    else:
        action = (0.00, 0.00)
        joint0goal = True



    if (joint0goal):
        if (not((angleJoint1 <= 0.017) and (angleJoint1 >= 0.0))):
            action = (0.00, 0.00011)
        else:
            action = (0.00, 0.00)

        

    observation, reward, terminated, truncated, info = env.step(action)
    xTarget = observation[4]
    yTarget = observation[5]

    xFingertip = observation[8] + xTarget
    yFingertip = observation[9] + yTarget
    print("FINGERTIP: ", xFingertip, yFingertip)
    print("TARGET: ", xTarget, yTarget)

    angleJoint0 = math.acos(observation[0])
    angleJoint1 = math.acos(observation[1])
    
    print("angle0: ", angleJoint0)
    print("angle1: ", angleJoint1)
    

    # if terminated or truncated:
        # observation, info = env.reset()

env.close()