import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('utils/MobileNetV2.h5')
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5
rect_bgr = (0, 0, 255)
img = np.zeros((500, 500))
text = 'text..'
text_width, text_height = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x, text_offset_y = 6, img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 2))

#cv2.rectangle(img, box_coords[0], box_coords[1], rect_bgr, cv2.FILLED)
#cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

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
        # Draw the status text on top of the face rectangle
        cv2.putText(frame, stat, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow('Face Mask Detection Task', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
