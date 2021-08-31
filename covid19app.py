import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np

covid19_model = keras.models.load_model('covid19model.h5')

def main():
    
    st.title("Detector de COVID19 basado de Deep Learning")
    col1, col2, col3 = st.columns(3)
    
    img_normal = Image.open("./ref_images/xray_normal.png")
    col1.subheader("NORMAL")
    col1.image(img_normal, use_column_width=True, caption='Radiografia de Tórax Normal')
        
    img_neumonia_virica = Image.open("./ref_images/xray_neumonia_virica.png")
    col2.subheader("NUEMONIA VIRICA")
    col2.image(img_neumonia_virica, use_column_width=True, caption='Radiografia de Tórax con Neumonia Vírica')
    
    img_covid19 = Image.open("./ref_images/xray_covid.png")
    col3.subheader("COVID19")
    col3.image(img_covid19, use_column_width=True, caption='Radiografia de Tórax con COVID19')
    
    image_object = st.file_uploader("Cargue la radiografía", type=["png", "jpg", "jpeg"])

    if image_object is not None:
        
        radiografia_img = Image.open(image_object)
        
        img_temp_file = 'radiografia-tmp.' + radiografia_img.format
        
        radiografia_img.save(img_temp_file)

        keras_img_object = keras.preprocessing.image.load_img(img_temp_file, target_size=(256, 256))
        
        img_array = keras.preprocessing.image.img_to_array(keras_img_object)
        img_array = img_array / 255.0
        img_array = img_array.reshape(-1, 256, 256, 1)
        
        predictions = covid19_model.predict(img_array)
        final_prediction_array = predictions[0]
        
        class_names = ['COVID-19', 'NORMAL', 'NEUMONIA-VÍRICA']
        
        prediccion = class_names[np.argmax(final_prediction_array)]
        probabilidad = 100 * np.max(final_prediction_array)
        
        st.image(radiografia_img, width=250)
        st.write(f'Predicción: {prediccion} - Probabilidad: {probabilidad:.4f}')
                       

if __name__ == '__main__':
    main()