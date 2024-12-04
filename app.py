from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

model = load_model('denememodel4_3.h5')
cv2.ocl.setUseOpenCL(False)
#emotion_dict = {0: "Kizgin", 1: "Mutlu", 2: "Notr", 3: "Uzgun"}
emotion_dict = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

@app.route('/predict', methods=['POST'])
def predict():
    # Gelen görüntüyü al
    image_data = request.files['image'].read()

    # Görüntüyü diziye dönüştür
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Görüntüyü griye dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    predictions = []

    for (x, y, w, h) in faces:
        # Yüzü kırp
        face_img = gray[y:y + h, x:x + w]

        # Yüzü yeniden boyutlandır ve normalleştir
        resized_img = cv2.resize(face_img, (48, 48))
        resized_img = resized_img / 255.0
        input_img = np.expand_dims(np.expand_dims(resized_img, -1), 0)

        # Tahmin yap
        prediction = model.predict(input_img)
        max_index = np.argmax(prediction[0])
        predicted_class = emotion_dict[max_index]
        predictions.append(predicted_class)

    # Tahmin sonucunu JSON formatında döndür
    if predictions:
        response = {'prediction': predictions[0]}  # İlk tahmini döndürür
    else:
        response = {'prediction': 'Yüz bulunamadı'}
    print(response)  # JSON yanıtını konsola yazdır

    return response

if __name__ == '__main__':
    app.run()