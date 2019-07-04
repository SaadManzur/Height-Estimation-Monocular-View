# Robust Height Estimation Using a Monocular Camera View

[![Robust Height Estimation](https://img.youtube.com/vi/9v9Y_5T8WBo/0.jpg)](https://www.youtube.com/watch?v=9v9Y_5T8WBo "Height Estimation")


Markerless motion capture, to my best knowledge, is an active research interest in current Computer Vision community. Even though multi-view motion capture has a richer dataset, estimating human pose from monocular camera has been receiving attention recently. But estimating motion from a single camera view is a fundamentally ill-posed problem, since multiple pose can generate similar images.

Since single camera does not provide enough information to project a 2D point, back to it’s original 3D coordinates, most of the monocular algorithm rely on geometric structure or information (i.e. vanishing points) present in the image. There are numerous work, which show methods to obtain reference frame from geometric cues of images. Once, the reference frame is known, by using relative ratios of lengths present within the image, we can obtain multiple features (height, gait etc.). However, the object has to be in contact with the reference plane to work.

In [_**Robust Estimation of Heights of Moving People Using a Single Camera**_](https://doi.org/10.1007/978-94-007-2911-7_36), the authors use a rectangular marker to generate a reference frame in the scene. By back projecting the 2D coordinates and intersecting them with the ground plane and global vertical line, they estimated height.

## Data Collection

To collect video dataset, I first tried to get the original video data, used in the paper. Since there was no references to the dataset used, I ended up making the dataset myself. Multiple videos were captured from multiple points around the campus. However, the video in front of the computer sciences building turned out to be best for the experiment. Since, the place has a relatively static background of the engineering building, moderate volume of traffic and tables to mount the camera. I first took a couple of shots to see where the marker can be located (even with a low resolution image). I opted for lower resolution to reduce the processing time.

For capturing the videos, I used a table mount to stabilize the camera and a Logitech C920 webcam. The video was captured at 640 × 480 resolution. The dataset can be found [here](https://www.youtube.com/watch?v=jsLipfHLZFM&list=PLVaC0E1yxiQ7ttgGKO5o4Tw7vf99pi2ot). When the marker was in the shadow, it could not detect sometimes. Hence, it was not used in the experiment.

## Methods

For camera calibration, I used a chess board pattern and took multiple images from different angles and used opencv provided camera calibration method to get the intrinsic matrix. The code can be found [here](https://github.com/SaadManzur/Research-Codes/tree/master/Camera/Calibration)

For this step, I generated a marker image and placed it on the reference plane. Since the dimension of sides of the markers are known (19.6 inch each), we have the 3d location of the four corners of the marker, assuming the origin is at center. Introducing z = 0, reduces the problem to estimating the homography between the reference plane and the image plane. So if we have 2d coordinates of the corners on the image plane (x , y) and actual 3d point (X, Y), we can estimate the homography using Zhang's method.

In the original paper, the authors used Kernel Density Estimator to do background subtraction. However, for my dataset, opencv provided MOG2 and KNN based background subtraction worked fine. After performing the background subtraction and some morphological operations, the foreground blobs were detected using opencv’s contour detection. Similarly, bounding rectangle was also extracted for the contours. Using the contours, the foreground points were isolated to perform principal component analysis to get eigen vectors and eigen values. The vertical eigen vector is used to find the intersection with the vertical bounds of the rectangle. These points are used as head and foot points on the image plane.

For a detailed procedured, refer to the pdf in the repository.
