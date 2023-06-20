import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Memuat model deteksi objek dari TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

# Menentukan lebar dan tinggi gambar yang akan diproses
width = 1028
height = 1028

# Memuat gambar menggunakan OpenCV
img = cv2.imread('image.jpg')

# Menyesuaikan ukuran gambar dengan input model
inp = cv2.resize(img, (width, height))

# Mengonversi gambar menjadi format RGB
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

# Opsional, tapi direkomendasikan (mengubah gambar menjadi tensor)
rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# Menambahkan dimensi tambahan pada tensor gambar
rgb_tensor = tf.expand_dims(rgb_tensor, 0)

# Menggunakan tensor gambar untuk melakukan prediksi label

# Menampilkan gambar asli
plt.figure(figsize=(10, 10))
plt.imshow(rgb)

# Mengeksekusi model deteksi objek pada tensor gambar
boxes, scores, classes, num_detections = detector(rgb_tensor)

# Memuat label kelas objek dari file CSV
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Mengambil label prediksi dari kelas yang dihasilkan
pred_labels = classes.numpy().astype('int')[0]
pred_labels = [labels[i] for i in pred_labels]

# Mengambil kotak pembatas, skor, dan label prediksi
pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

# Menampilkan kotak pembatas dan label prediksi
for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue

    score_txt = f'{100 * round(score)}%'
    img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (255, 255, 255), 2)

    # Mengukur ukuran teks untuk latar belakang
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
    (score_width, score_height), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Menghitung posisi dan ukuran latar belakang teks
    label_bg_coords = ((xmin, ymax - 10 - label_height - 10), (xmin + label_width, ymax - 10))
    score_bg_coords = ((xmax - score_width, ymax - 10 - score_height - 10), (xmax, ymax - 10))

    # Menggambar latar belakang teks
    cv2.rectangle(img_boxes, label_bg_coords[0], label_bg_coords[1], (255, 217, 61), -1)
    cv2.rectangle(img_boxes, score_bg_coords[0], score_bg_coords[1], (255, 217, 61), -1)

    # Menulis teks pada latar belakang
    cv2.putText(img_boxes, label, (xmin, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes, score_txt, (xmax - score_width, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# Menampilkan gambar hasil deteksi objek
plt.figure(figsize=(10, 10))
plt.imshow(img_boxes)

# Menyimpan gambar hasil deteksi objek
plt.savefig('image-result.jpg', transparent=True)