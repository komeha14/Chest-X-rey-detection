import subprocess
import sys
import streamlit as st
import numpy as np
from PIL import Image

# تحميل النموذج بطريقة مناسبة
# تأكد من طريقة التحميل وفقًا لنوع النموذج لديك

st.title('Chest X-ray Pneumonia Detection App')

# رفع صورة
uploaded_file = st.file_uploader("Upload a Chest X-ray image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    # معالجة الصورة
    image = image.resize((150, 150))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # إضافة بعد إضافي

    # هنا يجب أن تضع الكود الخاص بتنبؤ النموذج
    # prediction = model.predict(image)

    # استبدل بالنتيجة المناسبة
    prediction = np.random.rand(1, 1)  # فقط كمثال عشوائي

    if prediction[0][0] > 0.5:
        st.write("The model predicts: Pneumonia")
    else:
        st.write("The model predicts: Normal")
