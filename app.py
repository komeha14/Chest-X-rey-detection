import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# تحميل النموذج
model = load_model('model.h5')

# عنوان التطبيق
st.title('Chest X-ray Pneumonia Detection')

# تحميل الصورة
uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # تحويل الصورة إلى مصفوفة numpy ومعالجتها لتناسب النموذج
    img_array = np.array(image.resize((150, 150)))  # تعديل الحجم حسب ما يتوقعه النموذج
    img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد جديد للتوافق مع النموذج

    # توقع باستخدام النموذج
    prediction = model.predict(img_array)

    # عرض النتيجة
    if prediction[0][0] > 0.5:
        st.write("The model predicts this is Pneumonia.")
    else:
        st.write("The model predicts this is Normal.")
