import os, re, time

os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), ".cache")

import streamlit as st
import PIL
import cv2
import numpy as np
import uuid
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
from random import randint
from datetime import datetime

from src.colorify import device
from src.colorify.device_id import DeviceId
from src.colorify.visualize import *
from src.app_utils import get_model_bin


device.set(device=DeviceId.CPU)


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(model_dir, option):
    if option.lower() == 'artistic':
        model_url = 'https://www.dropbox.com/scl/fi/a20jp8dagw7y8nuzikq07/ColorizeArtistic_gen.pth'
        get_model_bin(model_url, os.path.join(model_dir, "ColorizeArtistic_gen.pth"))
        colorizer = get_image_colorizer(artistic=True)
    elif option.lower() == 'stable':
        model_url = "https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeStable_gen.pth?dl=0"
        get_model_bin(model_url, os.path.join(model_dir, "ColorizeStable_gen.pth"))
        colorizer = get_image_colorizer(artistic=False)

    return colorizer


def resize_img(input_img, max_size):
    img = input_img.copy()
    img_height, img_width = img.shape[0],img.shape[1]

    if max(img_height, img_width) > max_size:
        if img_height > img_width:
            new_width = img_width*(max_size/img_height)
            new_height = max_size
            resized_img = cv2.resize(img,(int(new_width), int(new_height)))
            return resized_img

        elif img_height <= img_width:
            new_width = img_height*(max_size/img_width)
            new_height = max_size
            resized_img = cv2.resize(img,(int(new_width), int(new_height)))
            return resized_img

    return img


def colorize_image(pil_image, img_size=800) -> "PIL.Image":
    # Open the image
    pil_img = pil_image.convert("RGB")
    img_rgb = np.array(pil_img)
    resized_img_rgb = resize_img(img_rgb, img_size)
    resized_pil_img = PIL.Image.fromarray(resized_img_rgb)

    # Send the image to the model
    output_pil_img = colorizer.plot_transformed_pil_image(resized_pil_img, render_factor=35, compare=False)
    
    return output_pil_img


def image_download_button(pil_image, filename: str, fmt: str, label="Download"):
    if fmt not in ["jpg", "png"]:
        raise Exception(f"Unknown image format (Available: {fmt} - case sensitive)")
    
    pil_format = "JPEG" if fmt == "jpg" else "PNG"
    file_format = "jpg" if fmt == "jpg" else "png"
    mime = "image/jpeg" if fmt == "jpg" else "image/png"
    
    buf = BytesIO()
    pil_image.save(buf, format=pil_format)
    
    return st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=f'{filename}.{file_format}',
        mime=mime,
    )


# STREAMLIT CODE 


st_color_option = "Artistic"

# Load models
try:
    with st.spinner("Loading..."):
        print('before loading the model')
        colorizer = load_model('models/', st_color_option)
        print('after loading the model')

except Exception as e: 
    colorizer = None
    print('Error while loading the model. Please refresh the page')
    print(e)
    st.write("**App loading error. Please try again later.**")



if colorizer is not None:
    st.title("Colorify - Shubham_Lad")

    st.image(open("assets/demo.jpg", "rb").read())

    st.markdown(
        """
        Colorizing black & white photo can be expensive and time consuming. So I am introducing to you an AI that can colorize
        grayscale photo in seconds. **Just upload your grayscale image, then click colorize.**
        """
    )
    
    uploaded_file = st.file_uploader("Upload photo", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img_input = PIL.Image.open(BytesIO(bytes_data)).convert("RGB")
        
        with st.expander("Original photo", True):
            st.image(img_input)

        if st.button("Colorify it!") and uploaded_file is not None:
            
            with st.spinner("Colorify is doing the magic!"):
                img_output = colorize_image(img_input)
                img_output = img_output.resize(img_input.size)
            
            # NOTE: Calm! I'm not logging the input and outputs.
            # It is impossible to access the filesystem in spaces environment.
            now = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            img_input.convert("RGB").save(f"./output/{now}-input.jpg")
            img_output.convert("RGB").save(f"./output/{now}-output.jpg")
            
            st.write("AI has finished the job!")
            st.image(img_output)
            # reuse = st.button('Edit again (Re-use this image)', on_click=set_image, args=(inpainted_img, ))
            
            uploaded_name = os.path.splitext(uploaded_file.name)[0]
            image_download_button(
                pil_image=img_output,
                filename=uploaded_name,
                fmt="jpg",
                label="Download Colorified Image"
            )
                
