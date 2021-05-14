import time
import edgeiq
import facial_landmarks
import cv2
"""
Real time facial landmarks using dlib 68 point predictor
"""


def main():

    text = "Facial Landmarks with Dlib"

    fps = edgeiq.FPS()

    shape_predictor = "shape_predictor_68_face_landmarks.dat"
    dlib_flm = facial_landmarks.Dlib_FLM(shape_predictor)

    try:
        with edgeiq.WebcamVideoStream(cam=0) as webcam, \
                edgeiq.Streamer() as streamer:
            # Allow webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = webcam.read()

                resized_frame, gray_resized_frame = dlib_flm.image_preprocessor(frame)
                facial_coordinates, rectangles = dlib_flm.detect_faces_shapes(gray_resized_frame)

                # Loop to markup resized_frame
                for(i, rectangle) in enumerate(rectangles):
                    (x, y, w, h) = dlib_flm.dlib_rectangle_to_cv_bondingbox(rectangle)
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                            resized_frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for facial_coordinate in facial_coordinates:
                    for (x, y) in facial_coordinate:
                        cv2.circle(resized_frame, (x, y), 1, (255, 0, 0), -1)

                streamer.send_data(resized_frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        # stop fps counter and display information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
