# socket json test

python에서 json을 어떻게 통신하는지 테스트해본 파일입니다.

`json_socket_receiver_test.ipynb`에서 소켓 서버를 열고,  

`json_socket_test.ipynb`에서 `test.json`을 가공한 다음 바이트 스트림으로 보내게 됩니다.

## 유의할 점

1. JSON을 보낼 시, str으로 변환한다.

1. str에 한글이 포함되어 있을 수 있으니 decode와 encdoe는 `utf-8`을 따른다.