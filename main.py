import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

cap = cv.VideoCapture(0)
model = load_model("C:/Users/Yoga/Desktop/PY/Models/mnist.h5")
while True:
    bool,frame = cap.read()
    if not bool:
        break

    img = cv.resize(frame,(28,28))
    img =cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    img = cv.Canny(img,50,100)
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img=img/255.0
    #print(img.size)
    pred = model.predict(img)
    #print(pred[0])
    #print(max(pred[0]))
    print(np.argmax(pred[0]))
    ans = np.argmax(pred[0])
    #cv.putText(frame,ans,(0,0),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv.imshow("Image", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
