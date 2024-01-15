import tkinter as tk
import tkinter.ttk
import threading
import pandas as pd  
import time 
import numpy as np 

import client 

POINTS = pd.read_csv('points.csv', index_col=0)
XYZ = POINTS[['x', 'y', 'z']].values 

MISSING_LEDS = [-1] + list(np.argwhere(np.isnan(XYZ).any(-1))[:, 0])


# Create a new instance of Tk
root = tk.Tk()
root.geometry('520x300')
root.title("XYZ Slider GUI")

# Create and pack three sliders for X, Y, and Z values
x_slider = tk.Scale(root, from_=-100, to=100, orient='horizontal', label='X')
x_slider.pack(fill='x')

y_slider = tk.Scale(root, from_=-100, to=100, orient='horizontal', label='Y')
y_slider.pack(fill='x')

z_slider = tk.Scale(root, from_=0, to=200, orient='horizontal', label='Z')
z_slider.set(100)
z_slider.pack(fill='x')


r_slider = tk.Scale(root, from_=1, to=100, orient='horizontal', label='R')
r_slider.set(25)
r_slider.pack(fill='x')

dropdown_var = tk.StringVar()
dropdown = tk.ttk.Combobox(root, textvariable=dropdown_var, values=MISSING_LEDS)
dropdown.pack()
dropdown.set(MISSING_LEDS[0])  # Set the default value of the dropdown


def colorize_from_distance(dists, r, color=(0, 0, 255), soft=False):
    if soft:
        strength = np.exp(-dists**2 / r**2)
    else:
        strength = np.where(dists < r, 1.0, 0.0)
    strength = np.where(np.isnan(strength), 0, strength)
    colors = strength * color 
    return colors.astype(np.uint8)

def colorize_leds_sphere(c, r, **kwargs):
    dists = np.linalg.norm(XYZ - c, axis=-1, keepdims=True)
    return colorize_from_distance(dists, r, **kwargs)
  
def colorize_leds_zplane(c, r, **kwargs):
    dists = abs(XYZ[:, 2] - c[2])[:, None]
    return colorize_from_distance(dists, r, **kwargs)

def update_leds():
    with client.LEDClient('192.168.4.1', nled=400) as juletre:
        while True:
            # Get the current values of the sliders
            x = x_slider.get() / 100
            y = y_slider.get() / 100
            z = z_slider.get() / 100
            r = r_slider.get() / 100

            led = int(dropdown_var.get())

            # Send these values to the API client
            colors = colorize_leds_zplane((x,y,z), r, soft=True)

            with juletre.update() as leds:
                leds[:] = colors 
                if led >= 0:
                    leds[led] = (255, 0, 0)
            # # api_client(x, y, z)
            # print(colors[:5])
            # # Wait for a short period before sending the next set of values
            # time.sleep(1)

# Start api thread 
thread = threading.Thread(target=update_leds)
thread.start()

# Start the main loop
root.mainloop()
update_leds()
import sys; sys.exit()
