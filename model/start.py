import predict
import cv2

img,happy,confidence = predict.starting('songdong.jpg')

print(happy,confidence)

cv2.imshow('Image view', img)
cv2.waitKey(0)
cv2.destroyAllWindows()