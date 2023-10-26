import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('utils/MobileNetV2.h5')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y: y + h, x: x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        faces = face_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in faces:
            face_roi = roi_color[ey: ey + eh, ex: ex + ew]

        final_im = cv2.resize(face_roi, (224, 224))
        final_im = np.expand_dims(final_im, axis=0)
        final_im = final_im / 255.0

        prediction = model.predict(final_im)
        print(prediction)

        if prediction < 1:
            stat = 'Mask Detected'
            text_color = (0, 255, 0)  # Green for mask detected
            frame_color = (0, 255, 0)
        else:
            stat = 'No Mask'
            text_color = (0, 0, 255)  # Red for no mask
            frame_color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
        cv2.putText(frame, stat, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow('Face Mask Detection Task', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
