import numpy as np 
import pandas as pd 
import cv2 
import colorsys 

import client 


POINTS = pd.read_csv('points.csv', index_col=0)
XYZ = POINTS[['x', 'y', 'z']].values - (0, 0, 0.5)
MISSING = np.isnan(XYZ).any(-1)[:, None]

FPS = 60
SPEED = 0.5 # m/s
DISTANCE = 2
SIGMA = 0.1


with client.LEDClient('192.168.4.1', nled=400) as juletre:

    phase = -DISTANCE / 2
    plane = None 
    color = None 
    num_planes = 2
    planes = []

    while True:
        with juletre.update() as leds:

            if not len(planes) or phase > DISTANCE / 2:
                print('Reset')
                planes = []
                for i in range(num_planes):
                    plane = np.random.normal(size=3)
                    plane = plane / np.linalg.norm(plane)
                    color = np.random.uniform(size=3)
                    color = color / color.max()
                    color = 200 * color 
                    planes.append((plane, color))
                phase = - DISTANCE / 2

            leds[:] = 0
            for plane, color in planes:
                dists = (XYZ * plane).sum(-1) - phase

                strength = np.exp(-dists**2 / SIGMA**2)
                strength = np.where(np.isnan(strength), 0, strength)

                col = strength[:, None] * color 
                col = np.clip(col, 0, 255).astype(np.uint8)
                leds[:] += col

            phase += SPEED / FPS