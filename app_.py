import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from copy import deepcopy

from src.core import process_inpaint


st.title("Colorify - Shubham_Lad")


st.markdown(
    """
    Colorizing black & white photo can be expensive and time consuming. So I am introducing to you an AI that can colorize
    grayscale photo in seconds. **Just upload your grayscale image, then click colorize.**
    """
)
uploaded_file = st.file_uploader("Choose image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img_input = Image.open(BytesIO(bytes_data)).convert("RGBA")

    if uploaded_file is not None and st.button("Colorize!"):
        
        with st.spinner("Colorify is doing the magic!"):
            img_output = """TODO"""
        
        # NOTE: Calm! I'm not logging the input and outputs.
        # It is impossible to access the filesystem in spaces environment.
        now = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        img_input.convert("RGB").save(f"./output/{now}.jpg")
        Image.fromarray(img_output).convert("RGB").save(f"./output/{now}-edited.jpg")
        
        st.write("Colorify has finished its assigned job!")
        st.image(img_output)
        # reuse = st.button('Edit again (Re-use this image)', on_click=set_image, args=(inpainted_img, ))
        
        with open(f"./output/{now}-edited.jpg", "rb") as fs:
            uploaded_name = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label="Download",
                data=fs,
                file_name=f'edited_{uploaded_name}.jpg',
            )
            
        st.info("**TIP**: If the result is not perfect, you can download then "
                "re-upload the result then remove the artifacts.")
