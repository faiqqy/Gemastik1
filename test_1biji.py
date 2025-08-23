import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import re
import aes
import qr


# --- KONFIGURASI PENTING ---
# 1. Path folder diatur ke dataset augmentasi Anda
TEST_IMAGE_FOLDER = 'Muhfaiq\\test1' #'dataset_augmented/test_data/'

# 2. Pastikan nama file model .h5 Anda sesuai

MODEL_FILES = {
    'Leukocytes': 'model_Leukocytes.keras',
    'Nitrite': 'model_Nitrite.keras',
    'Urobilinogen': 'model_Urobilinogen.keras',
    'Protein': 'model_Protein.keras',
    'pH': 'model_pH.keras',
    'Blood': 'model_Blood.keras',
    'Specific Gravity': 'model_Specific_Gravity.keras', # Menggunakan underscore
    'Ketone': 'model_Ketone.keras',
    'Bilirubin': 'model_Bilirubin.keras',
    'Glucose': 'model_Glucose.keras'
}

# 3. WAJIB ISI: Urutan kelas harus sama persis dengan saat training
LABEL_MAPS = {
    'Leukocytes': ['Neg.', 'Trace', 'Small', 'Moderate', 'Large'],
    'Nitrite': ['Neg.', 'Pos.'],
    'Urobilinogen': ['Normal', '16', '+', '++', '+++'], 
    'Protein': ['Neg.', 'Trace', '+', '++', '+++', '++++'],    
    'pH': ['5.0', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5'],         
    'Blood': ['Neg.', 'Non hemolyzed', 'Hemolyzed', '25 Small', '80 Moderate', '200 Large'],
    'Specific Gravity': ['1.000', '1.005', '1.010', '1.015', '1.020', '1.025', '1.030'], 
    'Ketone': ['Neg.', 'Trace', 'Small', 'Moderate', '8.0', '16'],      
    'Bilirubin': ['Neg.', 'Small', 'Moderate', 'Large'],   
    'Glucose': ['Neg.', 'Trace', '+', '++', '+++', '++++']
}
# -----------------------------

# Salin fungsi 'extract_features' yang sama persis dari script 'final_training.py' Anda
def extract_features(image_bgr, advanced=False):
    # ... (Salin fungsi lengkapnya persis dari script training/main_app.py) ...
    # Pastikan fungsi ini mengembalikan list berisi 30 atau 60 fitur
    try:
        if image_bgr is None: return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 15, 60) 
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        dipstick_crop = image_bgr[y:y+h, x:x+w]
        if dipstick_crop.size == 0: return None
        hsv_dipstick = cv2.cvtColor(dipstick_crop, cv2.COLOR_BGR2HSV)
        height, width, _ = hsv_dipstick.shape
        panel_width = width // 10
        if panel_width == 0: return None
        
        color_features = []
        for i in range(10):
            startX = i * panel_width
            endX = (i + 1) * panel_width
            roi = hsv_dipstick[int(height*0.2):int(height*0.8), startX:endX]
            if roi.size == 0: continue
            
            if advanced:
                mean, std_dev = cv2.meanStdDev(roi)
                features_per_panel = list(mean.flatten()) + list(std_dev.flatten())
                color_features.extend(features_per_panel)
            else:
                mean = cv2.mean(roi)
                color_features.extend(mean[:3])

        expected_length = 60 if advanced else 30
        return color_features if len(color_features) == expected_length else None
    except Exception:
        return None

# 1. Muat semua model Keras .h5
print("Memuat semua model Keras (.h5)...")
models = {}
for param, model_file in MODEL_FILES.items():
    try:
        models[param] = tf.keras.models.load_model(model_file, compile=False)
        print(f"Model untuk '{param}' berhasil dimuat.")
    except Exception as e:
        print(f"ERROR: Gagal memuat model '{model_file}': {e}")

def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Gagal memuat gambar."

    # Menggunakan fitur dasar (30) sesuai training terakhir Anda
    features = extract_features(image, advanced=False) 
    
    if features is None:
        return "Tidak dapat mengekstrak fitur."

    input_data = np.array([features], dtype=np.float32)
    results = {}

    for param, model in models.items():
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction[0])
        
        if param in LABEL_MAPS:
            predicted_label = LABEL_MAPS[param][predicted_index]
        else:
            predicted_label = f"Kelas Index {predicted_index}"
        
        results[param] = predicted_label
    
    return results

# --- SCRIPT UTAMA UNTUK ANALISIS INTERAKTIF ---
if __name__ == "__main__":

    img_src=analyze_image( 'Muhfaiq\\test1\\Foto (203).jpg')

    result = ",".join(f"{v}" for k, v in img_src.items())

    print(result)

    encrypt = aes.encryption(result,"opadfahadfladfaj")
    print(encrypt)

    qr.bikinQr(encrypt)