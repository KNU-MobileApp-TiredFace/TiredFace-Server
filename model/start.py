import predict
import cv2
import base64
import json


def make_json(img, happy, confidence):
    # 인코딩된 스트링을 저장할 객체
    encoded_string = None
    # 이미지 저장
    cv2.imwrite('Cropped_image.jpg', img)
    # 이미지를 다시 열고 (이것은 대훈이가 이렇게 하는게 된다고 함 꼼수인듯 ㄱㅇㄷ)
    with open("Cropped_image.jpg", "rb") as image_file:
        encoding_string = base64.b64encode(image_file.read())
    # print("길이 계산 ")

    # print(len(encoding_string))

    # print(type(encoding_string))

    # print(encoding_string[:100])

    send_msg_dict = {'value': happy, 'confidence': confidence,
                    'Image': str(encoding_string)}

    json_msg = json.dumps(send_msg_dict)

    print(happy, confidence)

    cv2.imshow('Image view', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return json_msg


# Cropped된 이미지와 상태, 피곤함 측정 결과 정도를 img,happy,confidence에 받아옴. 
img,happy,confidence = predict.starting('image.jpg')

json_msg = make_json(img,happy,confidence)

print(type(json_msg),len(json_msg))
