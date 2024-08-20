import cv2
import time
cpt = 0
maxFrames = 60

cap=cv2.VideoCapture(0)

while cpt < maxFrames:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    cv2.imshow("test window", frame) 
    cv2.imwrite("C:/Users/bleda/Desktop/projectx/veriseti_ve_goruntuler/images/ilacg%d.jpg" %cpt, frame)
    time.sleep(0.5)
    cpt += 1
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()