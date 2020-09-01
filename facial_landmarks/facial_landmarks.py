import dlib
import edgeiq
import numpy as np
import cv2


class Dlib_FLM(object):
    def __init__(self, shape_predictor, landmark_range=68):
        self.shape_predictor = shape_predictor
        self.landmark_range = landmark_range
        # http://dlib.net/python/index.html#dlib.get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
    def image_preprocessor(self, image):
        image = edgeiq.resize(image, width=500)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray_image
    def detect_faces_shapes(self, gray_image, up_sampling = 1):
        facial_coordinates = []
        rectangles = self.detector(gray_image, up_sampling)
        for rectangle in rectangles:
            _shape = self.predictor(gray_image, rectangle)
            facial_coordinate = self.shape_to_np(_shape)
            facial_coordinates.append(facial_coordinate)
        return facial_coordinates, rectangles
    def shape_to_np(self, shape, dtype="int"):
        coordinates = np.zeros((self.landmark_range, 2), dtype=dtype)
        for i in range(0, self.landmark_range):
            coordinates[i] = (shape.part(i).x, shape.part(i).y)
        return coordinates
    def dlib_rectangle_to_cv_bondingbox(self, rectangle):
        x = rectangle.left()
        y = rectangle.top()
        w = rectangle.right() - x
        h = rectangle.bottom() - y
        return (x, y, w, h)
