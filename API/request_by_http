import requests
import cv2

nome_imagem = input('Digite o nome da imagem (extensão jpg): ')

if nome_imagem.split('.')[1] == 'jpg':
    url = 'http://127.0.0.1:5000/api'
    img = cv2.imread(nome_imagem)
    _, img_encoded = cv2.imencode('.' + nome_imagem.split('.')[1], img)
    r = requests.post(url, data = img_encoded.tobytes(), headers = {'content-type': 'image/jpg'})
    print(r.json())
else:
    print("Digite apenas nas extensões solicitadas!")
