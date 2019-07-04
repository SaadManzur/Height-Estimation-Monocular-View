"""
Tested with Python 3.7 and OpenCV 4.1.0
"""

import cv2 as opencv
import numpy as np

MARKER_SIZE = 19.6

# Logitech C920, computed from calibration code separately
K = np.array([[662.25801378, 0., 294.92509514],
              [0., 663.05010424, 209.48385376],
              [0., 0., 1.]])


def locate_marker(marker_id, input_img, show=False):
    # input_img = opencv.resize(input_img, (int(input_img.shape[1] / 5), int(input_img.shape[0] / 5)))

    marker_dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_7X7_1000)

    corners, ids, rejected_img_points = opencv.aruco.detectMarkers(input_img, marker_dictionary)

    points3d = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, -MARKER_SIZE / 2],
                         [-MARKER_SIZE / 2, -MARKER_SIZE / 2]])

    if show:
        output = opencv.aruco.drawDetectedMarkers(input_img, corners, ids)

        for i in range(ids.shape[0]):
            if ids[i] == marker_id:
                counter = 0
                for corner in corners[i][0]:
                    coord = '(' + str(points3d[counter, 0]) + ', ' + str(points3d[counter, 1]) + ', 0)'
                    opencv.circle(output, (corner[0], corner[1]), 4, (255, 0, 0), -1)
                    opencv.putText(output, coord,
                                   (int(corner[0] - 20),
                                    int(corner[1] - (-1) ** (counter // 2) * 20)),
                                   opencv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))
                    counter = counter + 1

        opencv.imshow('marker', output)
        opencv.waitKey(10000)

    if len(corners) == 0:
        return None, None

    return corners[0][0], points3d


def get_extrinsic(img_points, points_3d):
    img_points_h = np.hstack((img_points, np.ones((4, 1))))
    normalized_img_points = np.linalg.inv(K) @ img_points_h.T

    H, mask = opencv.findHomography(points_3d, normalized_img_points.T[:, :2])
    H /= np.linalg.norm(H[:, 0])

    r1 = H[:, 0]
    r2 = H[:, 1]
    r3 = np.cross(r1, r2)

    t = (2 * H[:, 2]) / (np.linalg.norm(r1) + np.linalg.norm(r2))

    R = np.hstack((r1[:, np.newaxis], r2[:, np.newaxis], r3[:, np.newaxis]))

    U, E, Vt = np.linalg.svd(R)

    R = U @ Vt

    return R, t[:, np.newaxis]


def get_projection_matrix(rotation, translation):
    extrinsic = np.hstack((rotation, translation))
    return K @ extrinsic


def find_marker_in_frame(frame, prev_P):
    img_points, points_3d = locate_marker(20, frame)

    if img_points is None:
        return prev_P

    R, t = get_extrinsic(img_points, points_3d)

    P = get_projection_matrix(R, t)

    return P


def show_axis(input_img, P, thickness=20):
    axis_3d = np.vstack((np.identity(3) * 15, np.ones((1, 3))))
    axis_2d = P @ axis_3d

    origin_3d = np.vstack((np.zeros((3, 1)), np.ones((1, 1))))
    origin_2d = P @ origin_3d
    origin_2d /= origin_2d[2, 0]

    # opencv.circle(input_img, (int(origin_2d[0, 0]), int(origin_2d[1, 0])), 20, (128, 135, 12), -1)

    for i in range(3):
        color = np.zeros((3,))
        color[2 - i] = 255
        axis_2d[:, i] /= axis_2d[2, i]
        opencv.arrowedLine(input_img,
                           (int(origin_2d[0, 0]), int(origin_2d[1, 0])),
                           (int(axis_2d[0, i]), int(axis_2d[1, i])),
                           (color[0], color[1], color[2]),
                           thickness
                           )


def subtract_background_and_get_frames(filename, P = None):
    video = opencv.VideoCapture(filename)

    out_video = opencv.VideoWriter("output.avi",
                                   opencv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0,
                                   (640, 480))

    while not video.isOpened():
        video = opencv.VideoCapture(filename)

        if opencv.waitKey(1000) == 'q':
            break

        print("Attempting again.")

    print('Frame count = ', video.get(opencv.CAP_PROP_FRAME_COUNT))
    print('Frame width = ', video.get(opencv.CAP_PROP_FRAME_WIDTH))
    print('Frame height = ', video.get(opencv.CAP_PROP_FRAME_HEIGHT))

    subtractor = opencv.createBackgroundSubtractorMOG2(detectShadows=False)

    kernel = np.ones((9, 9), np.uint8)

    frames = []
    rects = []

    FRAME_SKIP = 200

    summation_P = np.zeros((3, 4))

    while True:
        ret, frame = video.read()

        if frame is not None:
            #P = find_marker_in_frame(frame, P)

            #if P is not None:
            #    summation_P = summation_P + P

            foreground_mask = subtractor.apply(frame)

            foreground_mask = opencv.morphologyEx(foreground_mask, opencv.MORPH_OPEN, np.ones((3, 3), np.uint8))
            foreground_mask = opencv.morphologyEx(foreground_mask, opencv.MORPH_CLOSE, kernel)

            contours, ret = opencv.findContours(foreground_mask, mode=opencv.CHAIN_APPROX_SIMPLE,
                                                method=opencv.RETR_FLOODFILL)

            if len(frames) % FRAME_SKIP == 0:
                opencv.imwrite('resources/frames/fg' + str(len(frames)) + '.jpg', foreground_mask)

            frame_contours = []
            for c in contours:
                area = opencv.contourArea(c)
                if 600 < area:
                    x, y, w, h = opencv.boundingRect(c)

                    if h > 1.5 * w:
                        frame_contours.append(np.array([x, y, w, h]))
                        opencv.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0))
                        head, foot = draw_head_and_foot(frame, np.array([x, y, w, h]), c)

                        if len(frames) % FRAME_SKIP == 0:
                            opencv.imwrite('resources/frames/hf' + str(len(frames)) + '.jpg', frame)

                        foot3d = estimate_foot(foot, P, frame)
                        head3d = estimate_head(head, P, frame, foot3d)
                        height = np.linalg.norm(head3d - foot3d)
                        ft_inch = str(int(height) // 12) + ' ft ' + str(int(height) % 12) + ' inch'
                        opencv.rectangle(frame,
                                         (int(head[0]), int(head[1]) - 20),
                                         (int(head[0]) + 100, int(head[1])),
                                         (255, 255, 255), -1)
                        opencv.putText(frame, ft_inch,
                                       (int(head[0]), int(head[1]) - 5),
                                       opencv.FONT_HERSHEY_SIMPLEX,
                                       0.5, (0, 0, 0), 1)

            if P is not None:
                show_axis(frame, P, thickness=2)

            opencv.rectangle(frame, (0, 0), (60, 20), (255, 255, 255), -1)
            opencv.putText(frame, str(len(frames)), (5, 15), opencv.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0))

            #draw_xy_plane(frame, P)

            if len(frames) % FRAME_SKIP == 0:
                opencv.imwrite('resources/frames/final' + str(len(frames)) + '.jpg', frame)

            out_video.write(foreground_mask)

            opencv.imshow('video', foreground_mask)
            opencv.waitKey(20)

            frames.append(frame)
            rects.append(frame_contours)

        else:
            break

    print("Background subtraction complete.")

    video.release()
    out_video.release()

    opencv.destroyAllWindows()

    return frames, rects, summation_P / len(frames)


def get_orientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = opencv.PCACompute2(data_pts, mean)

    return mean, eigenvectors, eigenvalues


def draw_line_through(img, p_, q_, color, rect):
    p = list(p_)
    q = list(q_)

    m = (q[0] - p[0]) / (q[1] - p[1])

    pt1 = (int(m * (rect[1] - p[1]) + p[0]), rect[1])
    pt2 = (int(m * (rect[1] + rect[3] - p[1]) + p[0]), rect[1] + rect[3])

    opencv.circle(img, pt1, 3, color, -1)
    opencv.circle(img, pt2, 3, color, -1)
    opencv.line(img, pt1, pt2, color, 1)

    return pt1, pt2


def draw_head_and_foot(frame, rect, contour):
    mean, eigen_vectors, eigen_values = get_orientation(contour)

    center = (int(mean[0, 0]), int(mean[0, 1]))

    p1 = (center[0] + 0.02 * eigen_vectors[0, 0] * eigen_values[0, 0],
          center[1] + 0.02 * eigen_vectors[0, 1] * eigen_values[0, 0])

    head, foot = draw_line_through(frame, center, p1, (0, 0, 255), rect)

    return head, foot


def estimate_foot(foot, projection_matrix, frame):
    P = projection_matrix[:, :3]
    p = projection_matrix[:, 3]

    inv_P = np.linalg.inv(P)

    C = - inv_P @ p

    z = np.zeros((1, 3))
    z[0, 2] = 1

    foot_h = np.array([np.array([foot[0], foot[1], 1])]).T

    l = - (z @ C[:, np.newaxis]) / (z @ inv_P @ foot_h)

    foot_org = C[:, np.newaxis] + l * (inv_P @ foot_h)

    projected_foot = projection_matrix @ np.vstack((foot_org, np.ones((1, 1))))
    projected_foot /= projected_foot[2, 0]

    opencv.circle(frame, (int(projected_foot[0, 0]), int(projected_foot[1, 0])),
                  4, (0, 0, 0), 2)

    return foot_org


def estimate_head(head, projection_matrix, frame, foot3d):
    P = projection_matrix[:, :3]
    p = projection_matrix[:, 3]

    inv_P = np.linalg.inv(P)

    C = - inv_P @ p

    Dv = np.zeros((1, 3))
    Dv[0, 2] = 1

    head_h = np.array([np.array([head[0], head[1], 1])]).T

    Dh = inv_P @ head_h

    B = foot3d - C[:, np.newaxis]
    A = np.hstack((Dh, -Dv.T))

    result = np.linalg.lstsq(A, B, rcond=None)[0]

    head_org = C[:, np.newaxis] + result[0] * Dh

    projected_head = projection_matrix @ np.vstack((head_org, np.ones((1, 1))))
    projected_head /= projected_head[2, 0]

    opencv.circle(frame, (int(projected_head[0, 0]), int(projected_head[1, 0])),
                  4, (0, 0, 0), 2)

    return head_org


def get_marker_corners_from_video(filename):
    video = opencv.VideoCapture(filename)

    while not video.isOpened():
        video = opencv.VideoCapture(filename)

        if opencv.waitKey(1000) == 'q':
            break

        print("Attempting again.")

    marker_dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_7X7_1000)

    points3d = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, -MARKER_SIZE / 2],
                         [-MARKER_SIZE / 2, -MARKER_SIZE / 2]])

    summation_corners = np.zeros((4, 2))
    counter = 0

    print("Estimating marker corners")
    while True:
        ret, frame = video.read()

        if frame is not None:
            corners, ids, rejected_img_points = opencv.aruco.detectMarkers(frame, marker_dictionary)

            if len(corners) is not 0:
                summation_corners += corners[0][0]
                counter += 1
        else:
            break

    video.release()

    opencv.destroyAllWindows()

    corners = summation_corners / counter

    print("Marker corners estimated")

    return corners, points3d


def draw_xy_plane(frame, projection_matrix):
    pts3d = np.array([[0, 20, 20, 0], [0, 0, 20, 20], [0, 0, 0, 0], [1, 1, 1, 1]])
    pts2d = projection_matrix @ pts3d

    pts2d /= pts2d[2, :]

    opencv.polylines(frame, [pts2d.astype(np.int32)[1:3, :].T], 1, (255, 0, 0), 2)


if __name__ == '__main__':
    filename = 'resources/ghost2.webm'

    img_points, points_3d = get_marker_corners_from_video(filename)

    R, t = get_extrinsic(img_points, points_3d)

    P = get_projection_matrix(R, t)

    frames, rects, projection_matrix = subtract_background_and_get_frames(filename, P)
