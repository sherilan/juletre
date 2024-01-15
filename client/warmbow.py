import numpy as np 
import pandas as pd 
import cv2 
import colorsys 

import client 
import utils 


POINTS = pd.read_csv('points.csv', index_col=0)
XYZ = POINTS[['x', 'y', 'z']].values 
MISSING = np.isnan(XYZ).any(-1)[:, None]

FPS = 60
SPEED = 0.5 # m/s
DISTANCE = 0.5
DIRECTION = (1, 0, 0)
CENTER = (0, 0, 0.75)

ang_vel = np.zeros(3, dtype=float)
direction = np.array(DIRECTION, dtype=float) / sum(DIRECTION)
center = np.array(CENTER, dtype=float)

def update_direction():
    global ang_vel, direction
    dt = 1 / FPS 
    max_ang_force = 0.02
    max_ang_speed = 0.05
    ang_vel = ang_vel + dt * max_ang_force * np.random.normal(size=3)
    ang_speed = np.linalg.norm(ang_vel)
    ang_vel = ang_vel * max_ang_speed / max(ang_speed, max_ang_speed)
    ang_vel_rot = utils.rotation_vector_to_matrix(ang_vel)
    direction = ang_vel_rot @ direction 
    



with client.LEDClient('192.168.4.1', nled=400) as juletre:

    phase = 0

    while True:
        with juletre.update() as leds:

            update_direction()
            # z = XYZ[:, 2]
            z = ((XYZ - center) * direction).sum(-1)
            prog = (z + phase) % DISTANCE
            hue = prog / DISTANCE 
            rgb = np.zeros((len(hue), 3))
            for i, h in enumerate(hue):
                if np.isnan(h):
                    continue 
                r = 150 + 50 * np.sin(h * 2 * np.pi)
                g = 35 + 25 * np.cos(h * 2 * np.pi)
                b = 0
                rgb[i] = (r, g, b)
            rgb = np.clip(rgb, 0, 255)
            rgb = np.where(MISSING, 0, rgb)
            leds[:] = rgb.astype(np.uint8)

            phase += SPEED / FPS 