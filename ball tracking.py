from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import time
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", help="path to video file")
ap.add_argument("-b","--buffer",type=int,default=64,help="max buffer size")
args = vars(ap.parse_args())

# Define the lower and upper bound for green color in the HSV color space
greenlower = (29, 86, 6)
greenupper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# Start video stream
if not args.get("video", False):
    vs = VideoStream(src=0).start()  # Use webcam
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)  # Allow time for camera to initialize

while True:
    # Grab the current frame
    frame = vs.read()
    if frame is None:
        break

    # Resize the frame and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the green color
    mask = cv2.inRange(hsv, greenlower, greenupper)

    # Perform a series of dilations and erosions to remove small blobs left in the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the center of the ball
    center = None

    # If there are any contours, find the largest one
    if len(contours) > 0:
        # Sort the contours by area and take the largest
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        
        # Calculate the center of the contour
        center = (x + w // 2, y + h // 2)

        # Draw a circle around the largest contour
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        # Draw a rectangle around the ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Update the points deque with the center of the ball
    pts.appendleft(center)

    # Draw the trail of the ball
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)

    # Show the frame with the tracked ball
    cv2.imshow("Tracked Ball", frame)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the windows
vs.stop() if not args.get("video", False) else vs.release()
cv2.destroyAllWindows()
