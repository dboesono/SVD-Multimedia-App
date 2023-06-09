import streamlit as st
import numpy as np
from src.ImageCompressor import image_functions
from src.ImageCompressor import compressor
from src.VideoBackgroundExtractor import extractor
from PIL import Image
from matplotlib.pyplot import imread
import tempfile
import os
import base64
from io import BytesIO
import time


def get_image_download_link(img, filename, text):
    """Generate download button in the streamlit app"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'''
    <a style="color: #ffffff; background-color: #3498db; padding: 10px; border-radius: 3px; text-decoration: none;" 
       onmouseover="this.style.textDecoration='underline';" 
       onmouseout="this.style.textDecoration='none';"
       href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'''
    return href


def main():
    st.title("Welcome to Our Multimedia Processing Web App")
    st.markdown("Choose an option in the sidebar to get started.")

    app_mode = st.sidebar.selectbox("Choose the app mode", ["Select a mode", "Image Compression", "Video Background Extraction"])

    # If use choose the image compression feature
    if app_mode == "Image Compression":
        st.header("Image Compression")
        st.markdown("Please upload your image and set the compression parameters.")

        # Set the compression parameter and the SVD algorithm
        k = st.sidebar.slider("Rank for SVD", 0, 250, 50) 
        svd_algo = st.selectbox('Select SVD Algorithm', ('NumPy', 'Version A', 'Version B'))

        # Implement your image compression functions here
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            # Continue with your image compression logic here...
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            image_array = image_functions.img2double(np.array(image))

            # Compress image with the specified type of SVD algorithm
            if svd_algo == "NumPy":
                compressed_image, compression_time, size_reduction = compressor.compress_svd(image_array, k, algo='numpy')
            elif svd_algo == "Version A":
                compressed_image, compression_time, size_reduction = compressor.compress_svd(image_array, k, algo='phase_A')
            elif svd_algo == "Version B":
                compressed_image, compression_time, size_reduction = compressor.compress_svd(image_array, k, algo='phase_B')

            # Output compressed image with size reduction and runtime
            compressed_image_pil = Image.fromarray(np.uint8(compressed_image))
            st.image(compressed_image_pil, caption="Compressed Image", use_column_width=True)
            st.write(f"Compression Time: {round(compression_time, 3)} seconds")
            st.write(f"Size Reduction: {size_reduction}%")
            st.markdown(get_image_download_link(compressed_image_pil, 'compressed_image.jpg', 'Download compressed image'), unsafe_allow_html=True)

    # If user choose the video background extraction feature
    elif app_mode == "Video Background Extraction":
        st.header("Video Background Extraction")
        st.markdown("Please upload your video file.")

        # Implement your video background extraction functions here
        uploaded_file = st.file_uploader("Choose a video file...", type=['mp4'])
        if uploaded_file is not None:
            # Continue with your video background extraction logic here...
            # Note: You may need to handle the uploaded_file appropriately before passing it to your function
            # It comes as a BytesIO object from the uploader. You might need to save it as a temp file first.
            tfile = tempfile.NamedTemporaryFile(delete=False)
            try:
                tfile.write(uploaded_file.read()) # tfile now holds the path to the temporary file which contains your uploaded data

                # Load video
                video_path = tfile.name
                video, height, width = extractor.input_video(video_path)
                
                # Start time
                start_time = time.time()

                # Extract background information matrix 
                B = image_functions.img2double(extractor.mixed_color(video_path))
                
                # Turn the background information matrix into an image
                background = Image.fromarray((B * 255).astype(np.uint8))
                
                # Output iamge
                st.image(background, caption="Extracted Video Background", use_column_width=True)

                # End time
                end_time = time.time()

                # Calculate elapsed time
                elapsed_time = end_time - start_time
                st.write(f"Background extraction time: {round(elapsed_time, 3)} seconds")     
                st.markdown(get_image_download_link(background, 'background_image.jpg', 'Download background image'), unsafe_allow_html=True)
            finally:
                # Now we're done with the file, so we can delete it
                tfile.close()
                os.unlink(tfile.name)


# Run the streamlit app
if __name__ == "__main__":
    main()