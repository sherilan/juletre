import cv2
import numpy as np 


class Camera:

    class Intrinsics:

        def __init__(self, mtx, dist):
            self.mtx = np.array(mtx).astype(float)
            self.dist = np.array(dist).astype(float)
            # Cache 
            self.undistort_params = {}
        
        def get_undistort_params(self, shape):
            if not shape in self.undistort_params:
                h, w = shape 
                self.undistort_params[shape] = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (w,h), 1, (w,h)
                )
            return self.undistort_params[shape]

        def undistort(self, image, method='simple', crop=True):
            h, w, _ = image.shape 
            newcameramtx, roi = self.get_undistort_params((h, w))
            if method == 'simple':
                image = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
            elif method == 'rectify_map':
                mapx, mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w,h), 5)
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) 
            else:
                raise ValueError(f'Unknown undistory method "{method}"')
            if crop:
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
            return image 

        @classmethod 
        def macbook_pro_2020(cls):
            return cls(
                mtx=[
                    [954.29569179,   0.,         648.32197358],
                    [  0.,         953.73692287, 351.22180692],
                    [  0.,           0.,           1.        ]
                ],
                dist=[ 0.11581714, -0.30598227, -0.00086317, -0.00063193,  0.32667599]
            )

    def __init__(self, index : int =0, intrinsics : Intrinsics =None):
        self.index = index 
        self.intrinsics = intrinsics
        self._cam = None 

    @property 
    def cam(self):
        if self._cam is None:
            raise RuntimeError('Cannot access camera outside context')
        return self._cam
    
    def capture(self, undistort=None, on_error='raise'):
        success, image = self.cam.read()
        if not success:
            raise RuntimeError(f'Failed to read from camera {self.index}')
        if undistort:
            if not self.intrinsics:
                raise ValueError('Cannot undistort without camera intrinsics')
            if isinstance(undistort, str):
                undistort = {'method': undistort}
            image = self.intrinsics.undistort(image, **undistort)
        return image 

    def __enter__(self):
        self._cam = cv2.VideoCapture(self.index)
        return self 
    
    def __exit__(self, exc, val, tb):
        try:
            self._cam.release()
        except: pass 
        self._cam = None 


if __name__ == '__main__':

    WIN_NAME = 'Camera'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    with Camera(index=0) as cam: #, intrinsics=Camera.Intrinsics.macbook_pro_2020()) as cam:
        while True:
            cv2.imshow(WIN_NAME, cam.capture())#undistort='simple'))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# # cap.set(3, frameWidth)
# # cap.set(4, frameHeight)
# # cap.set(10,150)


# while True:
#    success, img = cap.read()
   