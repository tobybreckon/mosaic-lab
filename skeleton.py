#####################################################################

# Example : real-time mosaicking - skelton outline of functionality

# takes input from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Base File Author : Toby Breckon, toby.breckon@durham.ac.uk

# Studenr Author(s) : <INSERT NAMES>

# Copyright (c) 2017 Dept. Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys
import numpy as np

#####################################################################

keep_processing = True;
camera_to_use = 0; # 0 if you have one camera, 1 or > 1 otherwise

#####################################################################

# define video capture object


cap = cv2.VideoCapture();

# define display window names

windowNameLive = "Live Camera Input"; # window name
windowNameMosaic = "Mosaic Output";

# initially our mosaic is an empty image

mosaic = None;

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((len(sys.argv) == 2) and (cap.open(str(sys.argv[1]))))
    or (cap.open(camera_to_use))):

    # create windows by name (as resizable)

    cv2.namedWindow(windowNameLive, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameMosaic, cv2.WINDOW_NORMAL);

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read();

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False;
                continue;

        # *** BEGIN outline of required mosaicking code ***

        # detect features in current image

        # if enough features present in image

            # if current mosaic image is empty (i.e. at start of process)

                # copy current frame to mosaic image

                # continue to next frame (i.e. next loop iteration)

            # else

                # get features in current mosaic (or similar)

                # (may need to check features are found, or can assume OK)

                # compute homography H between current image and mosaic

                # calculate the required size of the new mosaic image
                # if we add the current frame into it

                # merge the current frame into the new mosaic using
                # knowldge of homography H + required sise of image

                # (optional) - resize output mosaic to be % of full size
                # so it fits on screen or scale in porportion to screen size

        # else when not enough features present in image

            # continue to next frame (i.e. next loop iteration)

        mosaic = frame; # REMOVE this line (here so code runs at first)

        # *** END outline of required mosaicking code ***

        # display input and output (perhaps consider use of cv2.WND_PROP_FULLSCREEN)

        cv2.imshow(windowNameLive,frame);
        cv2.imshow(windowNameMosaic,mosaic);

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False;

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");

#####################################################################
