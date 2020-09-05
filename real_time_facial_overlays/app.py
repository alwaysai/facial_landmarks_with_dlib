import time
import edgeiq
import facial_landmarks
import numpy as np
import cv2
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull
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

imgEyePatch = cv2.imread('overlays/eye_patch.png',-1)
imgMustache = cv2.imread("overlays/mustache.png", -1)
imgGlass = cv2.imread("overlays/glasses.png", -1)

# Create the mask from the overlay image
orig_mask = imgEyePatch[:,:,3]
orig_mask_m = imgMustache[:,:,3]
orig_mask_g = imgGlass[:,:,3]

# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)
orig_mask_m_inv = cv2.bitwise_not(orig_mask_m)
orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)

# Convert the overlay image image to BGR
# and save the original image size
imgEyePatch = imgEyePatch[:,:,0:3]
origEyeHeight, origEyeWidth = imgEyePatch.shape[:2]

imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

imgGlass = imgGlass[:,:,0:3]
origGlassHeight, origGlassWidth = imgGlass.shape[:2]

def eye_size(eye):
    eyeWidth = dist.euclidean(eye[0], eye[3])
    hull = ConvexHull(eye)
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)
    eyeCenter = eyeCenter.astype(int)
    return int(eyeWidth), eyeCenter

def place_eye_patch(frame, eyeCenter, eyeSize):
   eyeSize = int(eyeSize * 1.5)

   x1 = int(eyeCenter[0] - (eyeSize/2))
   x2 = int(eyeCenter[0] + (eyeSize/2))
   y1 = int(eyeCenter[1] - (eyeSize/2))
   y2 = int(eyeCenter[1] + (eyeSize/2))

   h, w = frame.shape[:2]

   # check for clipping
   if x1 < 0:
     x1 = 0
   if y1 < 0:
     y1 = 0
   if x2 > w:
     x2 = w
   if y2 > h:
     y2 = h

   # re-calculate the size to avoid clipping
   eyeOverlayWidth = x2 - x1
   eyeOverlayHeight = y2 - y1

   # calculate the masks for the overlay
   eyeOverlay = cv2.resize(imgEyePatch, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)
   mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)
   mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)

   # take ROI for the overlay from background, equal to size of the overlay image
   roi = frame[y1:y2, x1:x2]

   # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

   # roi_fg contains the image pixels of the overlay only where the overlay should be
   roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)

   # join the roi_bg and roi_fg
   dst = cv2.add(roi_bg,roi_fg)

   # place the joined image, saved to dst back over the original image
   frame[y1:y2, x1:x2] = dst

def place_glasses(frame, glasses):
    temp1 = glasses[24][1]
    glassWidth = abs(glasses[16][0] - glasses[1][0]) #Face width where we need to place the glasses at eye level
    glassHeight = np.uint8(glassWidth * origGlassHeight / origGlassWidth)
    glass = cv2.resize(imgGlass, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    mask_g = cv2.resize(orig_mask_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    mask_g_inv = cv2.resize(orig_mask_inv_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    y1_g = np.uint8(temp1)
    y2_g = int(y1_g + glassHeight)
    x1_g = int(glasses[27][0] - (glassWidth/2))
    x2_g = int(x1_g + glassWidth)
    roi1 = frame[y1_g:y2_g, x1_g:x2_g]
    roi_g_bg = cv2.bitwise_and(roi1,roi1,mask = mask_g_inv)
    roi_g_fg = cv2.bitwise_and(glass,glass,mask = mask_g)
    frame[y1_g:y2_g, x1_g:x2_g] = cv2.add(roi_g_bg, roi_g_fg)

def place_mustache(frame, mustache):
    temp1 = mustache[33][1]
    temp2 = mustache[51][0]
    mustacheWidth = abs(3 * (mustache[31][0] - mustache[35][0])) #3 times the Nose width
    mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
    mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask_m = cv2.resize(orig_mask_m, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask_m_inv = cv2.resize(orig_mask_m_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    y1_m = int(temp1 - (mustacheHeight/2)) + 10
    y2_m = int(y1_m + mustacheHeight)
    x1_m = int(temp2 - (mustacheWidth/2))
    x2_m = int(x1_m + mustacheWidth)
    roi_m = frame[y1_m:y2_m, x1_m:x2_m]
    roi_bg_m = cv2.bitwise_and(roi_m,roi_m,mask = mask_m_inv)
    roi_fg_m = cv2.bitwise_and(mustache,mustache,mask = mask_m)
    frame[y1_m:y2_m, x1_m:x2_m] = cv2.add(roi_bg_m, roi_fg_m)

"""
Real time facial overlays using dlib 68 point predictor 
"""


def main():

    text = "Facial Overlays with Dlib"

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

                left_eye = 0
                right_eye = 0

                for facial_coordinate in facial_coordinates:
                    for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                        if name is 'left_eye':
                            left_eye = facial_coordinate[i:j]
                        #Uncoment if you want the patch on right eye as well.
                        #elif name is 'right_eye':
                        #    right_eye = facial_coordinate[i:j]
                    leftEyeSize, leftEyeCenter = eye_size(left_eye)
                    #Uncoment if you want the patch on right eye as well.
                    #rightEyeSize, rightEyeCenter = eye_size(right_eye)
                    place_mustache(resized_frame, facial_coordinate)
                    #Uncoment if you want to place spectacles on the face.
                    #place_glasses(resized_frame, facial_coordinate)
                    place_eye_patch(resized_frame, leftEyeCenter, leftEyeSize)
                    #place_eye(resized_frame, rightEyeCenter, rightEyeSize)

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
