import time
import cv2

COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

#Carregar as Classes/Tag    
class_names = []
with open("coco.names","r") as f:
    class_names  = [cname.strip() for cname  in f.readlines()]

#Carregar Video
cap = cv2.VideoCapture("Pessoa caminhando.mp4")

# Define o tamanho desejado para a janela
w = 1024
h = 514

# Define a janela de exibição das imagens, com tamanho automático
winName = 'TCC'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)


#Carregar informações da rede neural (peso e config.)
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

#Inputando parametros para o modelo da nossa rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416),scale=1/255)

#Loop
while True:

    _, frame = cap.read()

    start = time.time()

    # Detectar o item, o percentual e a desenhar a caixa
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    end = time.time()

    for(classid, score, box) in zip(classes, scores, boxes):

        #Criar uma cor para cada classe
        color = COLORS[int(classid) % len(COLORS)]
		
        if int(classid) == 0:
		
			#Texto em cima do quadrado e confiança
            label = f"{class_names[classid]} : {round(score*100)}%"

			#Desenho do retangulo
            cv2.rectangle(frame, box, color, 10)

			#Informações do retangulo: Na imagem = frame, Qual txt = label, Onde vai ficar = box.
            cv2.putText(frame, label, (box[0],box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)

    frame = cv2.resize(frame, (w, h))

    cv2.imshow("TCC", frame)

    # 27 p/ apartar ESC e sair, 60 para deixar o video na velocidade original
    if cv2.waitKey(60) == 27: 
        break # para interromper o video

cap.release()
cv2.destroyAllWindows()