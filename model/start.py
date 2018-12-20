import json
import socket
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import PIL
from datetime import datetime

import predict


PORT = 12321
S_PORT = 12322
HOST = '192.168.0.2'
BACKLOG = 10
PACKET_LEN = 1024 ** 2

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

    send_msg_dict = {'TiredScore': confidence,
                    'image': str(encoding_string)}

    json_msg = json.dumps(send_msg_dict)

    print(happy, confidence)

    # cv2.imshow('Image view', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return json_msg


if __name__=="__main__":
    while True:
        # 이미지 받기
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(BACKLOG)
        conn, addr = s.accept()
        print("[RECV]", datetime.now(), conn, addr)
        
        
        # 서버 다시열기
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind((HOST, S_PORT))
        s2.listen(BACKLOG)
        conn2, addr2 = s2.accept()
        print("[SEND]", datetime.now(), conn2, addr2)
        
        
        # 메세지 받기
        all_msg = ""
        while all_msg[-1:] != "}":
            while True:
                temp = conn.recv(PACKET_LEN)
                print(len(temp))
                all_msg += temp.decode('utf-8')
                if len(temp) < PACKET_LEN:
                    break

        conn.close()
        s.close()
        print("[RECV]", datetime.now(), "Done read msg")
            
        
        # 메세지 가공
        split_msg = all_msg.split('"')
        base_code = "".join(split_msg[11:-1])
        base64_decode = base64.b64decode(base_code)
        bytes_image = io.BytesIO(base64_decode)
        img = mpimg.imread(bytes_image, format="JPG")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("image.jpg", img)
        print("[IMG ]", datetime.now(), "Done image processing")
        
              
        # Cropped된 이미지와 상태, 피곤함 측정 결과 정도를 img,happy,confidence에 받아옴.
        img,happy,confidence = predict.starting('image.jpg')
        json_msg = make_json(img,happy,confidence)
        print("[IMG ]", datetime.now(), "Done predict")
        
        
        # 안드로이드 전송
        conn2.sendall(json_msg.encode('utf-8'))
        conn2.close()
        s2.close()
        print("[SEND]", addr, "Done")