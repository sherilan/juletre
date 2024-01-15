# juletre
Code dump for the 2024 Christmas Tree LED project

Inspired by: https://www.youtube.com/watch?v=v7eHTNm1YtU

Note1: I'm just dumping code that was hastily written here. It might need some rearranging/fixing of import paths to run properly. However, everything should be here (I think).

Note2: There are better libraries for controlling LEDs with an Arduino over wifi out there (e.g. WLED, although it doesn't work the same way), I just made this for fun and to learn.


## Overview

The system consists of an Arduino-based server that drives the LEDs, and a Python client for sending commands over wifi. I used an ESP32 board for the server, individually addressable WS2812 LEDs with the Neopixel arduino library, and a laptop to run the client.

### Server 

The server is provided in its entirety in the [esp32_tcp.ino](server/esp32_tcp.ino) Arduino sketch file.
It turns the Arduino into a wifi access point that another computer can connect directly to. 
It is also possible to connect it to an existing wifi network (which is way more convenient). Still, I found the direct access point approach to help maximize data throughput compared to joining our (not so great) home wifi network.
I also experimented with Bluethooth-based connections (both classic and BLE). It works alright for smaller setups (UP to 100 LEDs), but I could not find a way to make it reliable for 400 LEDs @ 60FPS. 

After setting up the access point, the esp32 sets up a simple TCP server on the wifi connection and initiates two concurrent tasks:

- A communication task that listens for a TCP connection and handles incoming data
- An illumination task that consumed instructions from the communication task and drives the LEDs with the NeoPixel library

To mitigate network jitter and get the 60Hz timing right, a ring buffer is deployed between the two tasks.
Each frame in the buffer contains RGB values for all the LEDs.
Whenever new data arrives over the TCP connection, the communication task will attempt to grab the next available buffer slot and populate it with the content of the message.
Meanwhile, the illumination task will consume the other side of the ring buffer, drive the LEDs, and then sleep to meet the targeted update rate. Once a buffer frame is displayed, a reply is given to the client to let it know that it can push more data (see Client section below).
As long as the client can push data at > 60Hz, this seems to work fine. If not, get a faster client or reduce the FPS in the Arduino sketch.

NOTE: The ESP32 has two cores. As far as I know, the underlying networking code that handles the low-level IP/TCP stuff runs on core 0 by default. After trying a bit of everything, I found that things work best when both tasks run on core 1.

### Client

The [client](client/client.py) is a python class that opens a TCP socket to the server and streams LED configurations. A very simple protocol is used right now; a header with meta data and a body with LED configuration. Right now, the only implemented type of message is "Dense", in which case each communication message involves dense RGB values (3 x N bytes) that will be displayed on the chain as-is (a bitmap essentially). You don't have to worry about FPS timing when using the client. Simply pump data into it as fast as you can and the communication protocol will ensure that it is displayed at 60Hz.

The client runs a dedicated listening thread for processing replies from the server. As mentioned above, each time a configuration is displayed on the LED chain, a reply is sent back to the client. This signals that a buffer slot has been consumed and the client can send more. This allows us to limit the amount of data the client can stream so that the server doesn't get overwhelmed. The limitation is implemented with a queue in the client. Before a command is sent to the server, the client stores a dummy variable in a queue. Whenever a reply is received from the server, an element is popped from the queue. The queue has a maximum size, which prevents the client from sending too many communication messages without receiving replies from the server. If too much data is pushed in, the main thread will simply block until the server acknowledges it.

### Calibration 

The setup above can be used to control the LEDs, but that only allows us to control the individual lights by their index along the chain.
To display cool, volumetric patterns, we need to know where the LEDs are located in 3D space.
This is done with classical computer vision. The overall procedure is as follows:

- Grab a camera and find its intrinsic parameters
- Take pictures of the LEDs with one and one light on from multiple angles
- Locate the position of the illuminated pixel in each image
- Get the 3D pose of all the cameras
- Recover the 3D position of each pixel by triangulation 

Essentially, for each LED, we get at least 2 pictures taken from different camera poses, "shoot out" rays corresponding to the brightest spot in the images, and see where those rays intersect.

#### Finding intrinsic parameters

The intrinsic parameters describe a specific camera according to the [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model).
Essentially, it allows us to map each pixel to a ray coming out of a point in 3D space.
There are several ways to get intrinsic parameters. Some cameras (e.g. realsense) come with software that allows us to lift them straight from an API.
For my laptop web camera, I couldn't find any such tool, so I estimated them empirically using a checkerboard and [tools from OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

#### Collecting images 

With a calibrated camera, we need to take a bunch of pictures of the LEDs from multiple vantage points.
Since we need to figure out where each LED is located in image space, we turn on one LED at a time and take a picture.
This is automated with the help of the LED client ([the data collection script](calib/collect_calib_data.py).

In my case, I called the data collection script 4 times. 
In all cases, I kept the camera in the same position but rotated the Christmas tree 90 degrees between each run. 

NOTE: turn the lights off in the room while doing this as it makes the next step a lot easier.

#### Finding the brightest pixel

Before figuring out where the LEDs are located in 3D world space, we need to find them in 2D image space.
Since I captured all the pictures in a dark room, this step is trivial and essentially boils down to argmaxing over a grayscale version of each image (using just the red channel since the LEDs were set to that color in the previous step).

#### Getting camera poses

To project 3D rays from each camera image, we need to know where the cameras are located in 3D space.
Defining the camera pose from the first set of images as the origin of the (arbitrary) world frame, this boils down to finding the relative transform from the first camera pose to all the other camera poses.
In theory, this transform can be estimated from just the pictures (by estimating [the essential matrix](https://en.wikipedia.org/wiki/Essential_matrix) from our known correspondences).
In practice, I found this approach to be too brittle and sensitive to noisy measurements (at least with the code I wrote).
Instead, I cheated a bit and simply calculated the camera poses by hand. 
As mentioned, the camera was located in a fixed pose relative to the tree (approx. 3 meters away and 1 meter up looking straight at the tree). 
Each 90-degree clockwise rotation of the tree around its vertical axis therefore corresponded to a 90-degree counter-clockwise orbit of the camera which can be calculated with straightforward trigonometry. 

#### Triangulating the 3D positions of LEDs

Knowing the intrinsic parameters of the camera and the different poses (extrinsic parameters) corresponding to each image, we can estimate the 3D positions of the LEDs.
For each image, we shoot a ray from the origin of the camera that goes through the brightest brightest pixel and then continues forever.
Using the rays from (at least) two images with the same LED activated, we find the point of (approximate) intersection.
This can be expressed as a least square regression problem.

This method is implemented in [the triangulation script](calib/triangulate.py). 
For each LED, it looks at the brightest pixel coordinates from the four different camera poses. It then selects the two images with the best pixel coordinates (in terms of recorded brightness) and triangulates the LED position. If no two images have brightness above a given threshold, it ignores that LED.

In my case, this worked for getting good coordinates for most LEDS. When displaying 3D patterns, I simply kept the ones that weren't properly calibrated off. Better (and maybe more) images would have helped to estimate the position of 100% of the LEDs.

