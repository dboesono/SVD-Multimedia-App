Image Compression & Video Background Extraction using SVD
==============================


## Project Overview

This project was undertaken as part of the course DDA3005: Numerical Methods during Term 1, 2022/2023 at The Chinese University of Hong Kong, Shenzhen (CUHK-SZ). The principal goal of the project is to explore, analyze, and implement various algorithms for computing singular value decompositions (SVD), and leverage SVD for diverse imaging applications. The project comprises four primary components:

1. Design and Implementation of the SVD Algorithm:
   Implemented the two-phase procedure, detailed in the lectures, to compute the singular value decomposition of any given matrix $\text{A} \in \mathbb{R}^{m \times n}$, denoted as:

   $$\text{A} = \text{U} \Sigma \text{V}^\text{T}$$

   This was achieved utilizing Python, exhibiting a solid understanding of linear algebra and programming.

2. Application of SVD in Image Deblurring:
   Explored the use of SVD in the context of image deblurring problems. This exploration aimed at identifying the most efficient SVD algorithm for image compression. We conducted a comparative analysis of the performance, accuracy of singular values, and the speed of convergence across a minimum of three different scenarios/images.

3. Utilization of SVD in Video Background Extraction:
   Leveraged SVD and power iterations to extract background information from given video data. We compared the efficiency and runtime of our bespoke algorithm against industry-standard software packages, demonstrating proficiency in implementing and optimizing numerical methods.

4. Development of a Web Application:
   Constructed a web application using the Streamlit framework. The application offers image compression and video background extraction functionalities based on the findings and methodologies derived from previous steps, showcasing ability to deploy machine learning models in a user-friendly application.

For a comprehensive understanding of the project, the complete project description is available [here](./reports/project_description.pdf).


## Team Members

| Student ID | Student Name   | Email                        
| :----------: | :--------------: | :------------------------------: | 
| 120040022  | Darren Boesono        | 120040022@link.cuhk.edu.cn     | 
| 120090356  | Yitao Wang        |    120090356@link.cuhk.edu.cn  | 
| 120040002  | Joseph Ariel Christopher Teja          | 120040002@link.cuhk.edu.cn     | 
| 120040023  | Jefferson Joseph Tedjojuwono            | 120040023@link.cuhk.edu.cn     | 


Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or 
    ├── README.md          <- The top-level README for developers using this project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── resources            
    │    ├── demo
    │    │    ├── jelly_fish_compression.gif   <- Image compression demo
    │    │    └── street_extraction.gif        <- Background extraction demo
    │    ├── test_images
    │    └── test_videos      
    │   
    │
    ├── notebooks                       <- Jupyter notebooks
    │    |                
    │    └──  analyze_results.ipynb     <- Results analytics in a jupyter notebook          
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                       <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   ├── project_description.pdf   <- Description about the project
    │   └── figures                   <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── ImageCompressor             <- Source code for image compression feature of the app
    │   │    │
    │   │    ├── __init__.py            <- Makes ImageCompressor a Python Module
    │   │    ├── image_functions.py     <- Contains Python functions for imaging applications
    │   │    └── compressor.py          <- Contain the image compressor for the app
    │   │
    │   ├── SVD                         <- Source for the main 2-Phase SVD algorithms
    │   │   │
    │   │   ├── __init__.py             <- Makes SVD a Python module
    │   │   └── two_phase_svd.py        <- The main 2-Phase SVD algorithms
    │   │
    │   └── VideoBackgroundExtractor    <- Source code for the video background extraction feature of the app
    │       │                 
    │       ├── __init__.py             <- Makes VideoBackgroundExtractor a Python Module
    │       └── extractor.py            <- Main code for the Video Background Extractor
    │   
    │
    ├── tox.ini     <- tox file with settings for running tox; see tox.readthedocs.io
    └── app.py      <- Main streamlit web application


## Installation
1. Clone this repository using the following command:
   
   ```bash
   git clone https://github.com/dboesono/SVD-ImageCompressor-VideoExtractor.git
   ```
2. Navigate to the project directory using the command:
   
   ```bash
   cd SVD-ImageCompressor-VideoExtractor
   ```
3. Create a virtual environment using the command:

   ```bash
   python -m venv env
   ```
4. Activate the virtual environment according to your operating system:
   - On Windows:
  
        ```bash
        .\env\Scripts\activate
        ``` 
    - On Unix or MacOS:
  
        ```bash
        source env/bin/activate
        ``` 
5. Install the necessary packages:
   
   ```bash
   pip install -r requirements.txt
   ```


## Usage
Run the following command to start the Streamlit web app:
```bash
streamlit run app.py
```


## Demo
Below are the video demos of the Image Compression and Video Background Extraction functionalities:


### Image Compression
![Image Compression Demo Video](https://github.com/dboesono/SVD-ImageCompressor-VideoExtractor/blob/main/resources/demo/jellyfish_compression.gif?raw=true)

### Video Background Extraction

We want to extract the background information of the video data with filename `street.mp4` which you can view [here](https://www.youtube.com/watch?v=OJR3K1VanFs):

![Video Background Extraction Demo](https://github.com/dboesono/SVD-ImageCompressor-VideoExtractor/blob/main/resources/demo/street_extraction.gif?raw=true)
