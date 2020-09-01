import edgeiq
import facial_landmarks
import numpy as np
import cv2
from collections import OrderedDict


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])


def main():
    text = "Facial Part Detection with Dlib"
    try:
        shape_predictor = "shape_predictor_68_face_landmarks.dat"
        dlib_flm = facial_landmarks.Dlib_FLM(shape_predictor)

        image_paths = sorted(list(edgeiq.list_images("images/")))
        print("Images:\n{}\n".format(image_paths))

        with edgeiq.Streamer(queue_depth=len(image_paths), inter_msg_time=3) as streamer:
            for image_path in image_paths:
                image = cv2.imread(image_path)
                resized_image, gray_image = dlib_flm.image_preprocessor(image)
                facial_coordinates, rectangles =dlib_flm.detect_faces_shapes(gray_image)

                for facial_coordinate in facial_coordinates:
                    for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                        print(name)
                        clone = resized_image.copy()
                        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 0), 2)
                        for (x, y) in facial_coordinate[i:j]:
                            cv2.circle(clone, (x, y), 3, (255, 0, 0), -1)
                            streamer.send_data(clone, text)
                        streamer.wait()
    finally:
         print("Program Ending")


if __name__ == "__main__":
    main()
