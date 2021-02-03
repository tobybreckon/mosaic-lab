#####################################################################

# Example : real-time mosaicking - skeleton outline of functionality

# takes input from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Base File Author : Toby Breckon, toby.breckon@durham.ac.uk

# Student Author(s) : <INSERT NAMES>

# Copyright (c) <YEAR> <INSERT NAMES>, Durham University, UK
# Copyright (c) 2017/18 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys

#####################################################################

# import all the provided helper functions

import mosaic_support as ms

#####################################################################

# check OpenCV version and if extra modules are present

print("OpenCV: " + cv2.__version__)
print("OpenCV Extra Modules Present: " + str(ms.extra_opencv_modules_present()))
print("OpenCV Non-Free Algorithms Present: " +
      str(ms.non_free_algorithms_present()))
print("Python: " + sys.version)
print()

#####################################################################

keep_processing = True
camera_to_use = 0  # 0 if you have one camera, 1 or > 1 otherwise

#####################################################################

# define video capture object

cap = cv2.VideoCapture()

# define display window names

window_name_live = "Live Camera Input"  # window name
window_name_mosaic = "Mosaic Output"

# initially set our mosaic to an empty image

mosaic = None

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((len(sys.argv) == 2) and (cap.open(str(sys.argv[1]))))
        or (cap.open(camera_to_use))):

    # create windows by name (as resizable)

    cv2.namedWindow(window_name_live, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_mosaic, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # TODO - insert some mechanism to take very Nth frame only

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

        # *** BEGIN TODO - outline of required mosaicking code ***

        # detect features in current image

        # if enough features present in image

            # if current mosaic image is empty (i.e. at start of process)

                # copy current frame to mosaic image

                # continue to next frame (i.e. next loop iteration)

            # else

                # get features in current mosaic (or similar)
                # (may need to check features are found, or can assume OK)

                # compute matches between current image and mosaic
                # (cv2.drawMatches() may be useful for debugging here)

                # compute homography H between current image and mosaic

                # calculate the required size of the new mosaic image
                # if we add the current frame into it

                # merge the current frame into the new mosaic using
                # knowldge of homography H + required sise of image

                # (optional) - resize output mosaic to be % of full size
                # so it fits on screen or scale in porportion to screen size

        # else when not enough features present in image
            # (cv2.drawKeypoints() may be useful for debugging here)

            # continue to next frame (i.e. next loop iteration)

        if (mosaic is None):  # *** TODO REMOVE this part ***
            mosaic = frame  # only here so code runs at first time

        # *** END TODO outline of required mosaicking code ***

        # display input and output (perhaps consider use of
        # cv2.WND_PROP_FULLSCREEN)

        cv2.imshow(window_name_live, frame)
        cv2.imshow(window_name_mosaic, mosaic)

        # start the event loop - wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        
        key = cv2.waitKey(40) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
