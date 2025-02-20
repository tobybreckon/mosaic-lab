#####################################################################

# Example : real-time mosaicking - skeleton outline of functionality

# takes input from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Base File Author : Toby Breckon, toby.breckon@durham.ac.uk

# Student Author(s) : <INSERT NAMES>

# Copyright (c) <YEAR> <INSERT NAMES>, Durham University, UK
# Copyright (c) 2017-25 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys
import argparse

#####################################################################

# import all the provided helper functions

import mosaic_support as ms

#####################################################################

# check OpenCV version and if extra modules are present

print("\nOpenCV: " + cv2.__version__)
print("OpenCV Extra Modules Present: " +
      str(ms.extra_opencv_modules_present()))
print("OpenCV Non-Free Algorithms Present: " +
      str(ms.non_free_algorithms_present()))
print("Python: " + sys.version)
print()

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform ' +
    sys.argv[0] +
    ' operation on incoming camera/video image')
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-r",
    "--rescale",
    type=float,
    help="rescale image by this factor",
    default=1.0)
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

#####################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not (args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window names

window_name_live = "Live Camera Input"  # window name
window_name_mosaic = "Mosaic Output"

# initially set our mosaic to an empty image

mosaic = None

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create windows by name (as resizable)

    cv2.namedWindow(window_name_live, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_mosaic, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

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

        # start the event loop - wait 500ms (i.e. 1000ms / 2 fps = 500 ms)

        key = cv2.waitKey(500) & 0xFF

        # detect specific key strokes
        # "x" = exit; "f" = fullscreen

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_name_mosaic,
                cv2.WND_PROP_FULLSCREEN,
                float(not (cv2.getWindowProperty(window_name_mosaic,
                                                 cv2.WND_PROP_FULLSCREEN))))

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
