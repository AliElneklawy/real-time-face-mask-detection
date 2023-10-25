import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('utils/final_model.h5')
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5
rect_bgr = (255, 255, 255)
img = np.zeros((500, 500))
text = 'text..'
text_width, text_height = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x, text_offset_y = 10, img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

cv2.rectangle(img, box_coords[0], box_coords[1], rect_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces = face_cascade.detectMultiScale(roi_gray)

        if len(faces) == 0: ('No faces')
        else:
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey: ey + eh, ex: ex + ew]
    
    final_im = cv2.resize(face_roi, (224, 224))
    final_im = np.expand_dims(final_im, axis=0)
    final_im = final_im / 255.0
    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5

    prediction = model.predict(final_im)
    print(prediction)
    if prediction > 0:
        stat = 'No Mask'
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, stat, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, stat, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
    else:
        stat = 'Mask Detected'
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, stat, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv2.putText(frame, stat, (100, 150), font, 3, (0, 255, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
    
    cv2.imshow('Face Mask Detection Task', frame)
    #cv2.waitKey(0)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
