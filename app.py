import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# تحميل النموذج
model = load_model('model.h5')

# عنوان التطبيق
st.title('Chest X-ray Pneumonia Detection App')

# رفع الصورة من المستخدم
uploaded_file = st.file_uploader("Upload a Chest X-ray image (JPG/PNG)", type=["jpg", "png", "jpeg"])

# التأكد من أن المستخدم رفع صورة
if uploaded_file is not None:
    # فتح الصورة باستخدام Pillow
    image = Image.open(uploaded_file)

    # عرض الصورة على التطبيق
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    # معالجة الصورة لتناسب المدخلات المطلوبة للنموذج
    image = image.resize((150, 150))  # التأكد من أن حجم الصورة مناسب للنموذج
    image = np.array(image)
    image = image / 255.0  # التطبيع
    image = np.expand_dims(image, axis=0)

    # توقع النموذج
    prediction = model.predict(image)

    # عرض النتيجة على المستخدم
    if prediction[0][0] > 0.5:
        st.write("The model predicts: Pneumonia")
    else:
        st.write("The model predicts: Normal")
