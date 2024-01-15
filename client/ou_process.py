import numpy as np 
import pandas as pd 
import cv2 
import colorsys 

import client 

center = np.array([0, 0, 0.5])

class Ball:

    def __init__(
            self, 
            color,
            center=(0, 0, 0.5), 
            sigma=0.2,
            force_restore=0.5,
            force_perturb=1.0,
            force_friction=1.0,
        ):
        self.color = np.array(color).astype(float)
        self.center = np.array(center).astype(float)
        self.sigma = sigma 
        self.force_restore = force_restore
        self.force_perturb = force_perturb
        self.force_friction = force_friction
        self.reset()

    def reset(self):
        self.pos = self.center.copy()
        self.speed = np.zeros(3)
    
    def update(self, dt):
        perturb = self.force_perturb * np.random.normal(size=3)
        restore = self.force_restore * (self.center - self.pos)
        friction = self.force_friction * (-self.speed)
        self.speed += dt * (perturb + restore + friction)
        self.pos += dt * self.speed 

    def paint(self, xyz):
        dists = np.linalg.norm(xyz - self.pos, axis=-1)
        strength = np.exp(-dists**2 / self.sigma**2)
        strength = np.where(np.isnan(strength), 0, strength)
        return strength[:, None] * self.color
        

        


POINTS = pd.read_csv('points.csv', index_col=0)
XYZ = POINTS[['x', 'y', 'z']].values - (0, 0, 0.5)
MISSING = np.isnan(XYZ).any(-1)[:, None]
FPS = 30

balls = [
    Ball((0, 0, 1)),
    Ball((0, 1, 0)),
    Ball((1, 0, 0)),
]

balls = [
    Ball(0.2 * 0.8 * np.random.uniform(size=3), sigma=0.15)
    for _ in range(10)
]

with client.LEDClient('192.168.4.1', nled=400) as juletre:


    while True:
        with juletre.update() as leds:
            colors = np.zeros_like(leds).astype(float)
            for ball in balls:
                ball.update(dt=1/FPS)
                colors[:] += ball.paint(XYZ)
            colors = np.clip(colors * 256, 0, 256).astype(np.uint8)
            leds[:] = colors