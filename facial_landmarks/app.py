import edgeiq
import facial_landmarks
import cv2


def main():
    text = "Facial Landmarks with Dlib"
    try:
        shape_predictor = "shape_predictor_68_face_landmarks.dat"
        dlib_flm = facial_landmarks.Dlib_FLM(shape_predictor)

        image_paths = sorted(list(edgeiq.list_images("images/")))
        print("Images:\n{}\n".format(image_paths))

        with edgeiq.Streamer(queue_depth=len(image_paths), inter_msg_time=3) as streamer:
            for image_path in image_paths:
                image = cv2.imread(image_path)
                image, gray_image = dlib_flm.image_preprocessor(image)
                facial_coordinates, rectangles = dlib_flm.detect_faces_shapes(gray_image)

                # Loop to markup image
                for(i, rectangle) in enumerate(rectangles):
                    (x, y, w, h) = dlib_flm.dlib_rectangle_to_cv_bondingbox(rectangle)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                            image, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for facial_coordinate in facial_coordinates:
                    for (x, y) in facial_coordinate:
                        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

                streamer.send_data(image, text)
            streamer.wait()
    finally:
        print("Program Ending")


if __name__ == "__main__":
    main()
