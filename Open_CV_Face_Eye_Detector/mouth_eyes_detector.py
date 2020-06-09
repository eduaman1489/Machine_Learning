import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('/Users/heisenberg/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/heisenberg/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)   # -1 is used if 3rd party webcam is used, for default it's 0
while True:
    # img = cv2.imread('lala.JPG')
    _, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 'Face', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
            #cv2.putText(img, 'Eyes', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

    #plt.imshow(img)
    #plt.show()
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()