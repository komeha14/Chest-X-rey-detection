import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# تحميل الموديل
model = load_model('model.h5')

# عنوان التطبيق
st.title("Chest X-ray Pneumonia Detection")

# تحميل صورة الأشعة السينية من المستخدم
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)
    
    # معالجة الصورة لتناسب النموذج (يفترض أن الصورة لها حجم محدد مثل 224x224)
    image = image.resize((224, 224))  # تعديل الحجم إذا لزم الأمر
    img_array = np.array(image) / 255.0  # تطبيع الصورة
    img_array = np.expand_dims(img_array, axis=0)  # تحويل الصورة إلى الشكل المناسب للنموذج
    
    # توقع النتيجة باستخدام الموديل
    prediction = model.predict(img_array)
    
    # عرض النتيجة
    if prediction[0][0] > 0.5:
        st.write("Prediction: Pneumonia Detected")
    else:
        st.write("Prediction: Normal")