import time 
import os 
import pathlib 
import numpy as np 
import cv2 

import client 
import camera 

# 3m + 0.52m away 
# 0.73 m + 0.25 m up


DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'calibration'
WIN_NAME = 'Camera'
NUM_LEDS = 400 
NAME = 'tree2'
VIEW = 4

saveto = DATA_PATH / NAME / str(VIEW) 
os.makedirs(saveto, exist_ok=True)


led = client.LEDClient('192.168.4.1', nled=NUM_LEDS)
cam = camera.Camera()

cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

with cam:
    time.sleep(1)
    for i in range(NUM_LEDS):
        # print(led._sock)
        with led.update() as rgbs:
            rgbs[:] = 0
            rgbs[i] = (100, 0, 0)#(100)
        print("OK", i)
        time.sleep(0.1)
        image = cam.capture()
        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(1)
        cv2.imwrite(str(saveto / f'img_{i}.png'), image)
        print('NEXT')


