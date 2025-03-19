import cv2

# MOG背景差分器
# https://www.jb51.net/article/236746.htm

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"D:\mp4\ISYAM_Smoke_Output.avi")
cap = cv2.VideoCapture(r"D:\tmp\outdoor_night_10m_gasoline_CCD_002.avi")
success, frame = cap.read()
while success:
    frame = cv2.resize(frame, (640, 480))
    fg_mask = bg_subtractor.apply(frame)

    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    if OPENCV_MAJOR_VERSION >= 4:
        # OpenCV 4 or a later version is being used.
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        # OpenCV 3 or an earlier version is being used.
        # cv2.findContours has an extra return value.
        # The extra return value is the thresholded image, which is
        # unchanged, so we can ignore it.
        _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('mog', fg_mask)
    cv2.imshow('thresh', thresh)
    cv2.imshow('detection', frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break

    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
