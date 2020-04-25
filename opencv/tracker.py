import cv2
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

trackers = {
    "BOOSTING": cv2.TrackerBoosting_create(),
    "MIL": cv2.TrackerMIL_create(),
    "KCF": cv2.TrackerKCF_create(),
    "TLD": cv2.TrackerTLD_create(),
    "MEDIANFLOW": cv2.TrackerMedianFlow_create(),
    "GOTURN": cv2.TrackerGOTURN_create(),
    "MOSSE": cv2.TrackerMOSSE_create(),
    "CSRT": cv2.TrackerCSRT_create(),
}

tracker = trackers["CSRT"]

# Read video
video = cv2.VideoCapture(0)

while True:
    ok, frame = video.read()
    mask = np.full_like(frame, 150, dtype=np.uint8)
    mask[165:315, 245:395] = 0
    background = cv2.add(frame, mask)  # create the rect frame with blurring background

    cv2.imshow(None, background)
    k = cv2.waitKey(60) & 0xFF
    if k == 32:  # it triggers tracking
        break


# Read first frame.
ok, frame = video.read()
bbox = (245, 165, 150, 150)  # x, y, w, h of bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if ok:

        # Update tracker
        ok, bbox = tracker.update(frame)

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(
                frame,
                "Tracking failure detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

        # Display tracker type on frame
        cv2.putText(
            frame,
            "CSRT" + "Tracker",
            (100, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 170, 50),
            2,
        )

        # Display FPS on frame
        cv2.putText(
            frame,
            "FPS : ",
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 170, 50),
            2,
        )

        # Display result
        cv2.imshow(None, frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
cv2.destroyAllWindows()
video.release()
