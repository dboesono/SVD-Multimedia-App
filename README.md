Image Compression & Video Background Extraction using SVD
==============================


## Project Overall Description

This project is dedicated to the course project of the course DDA3005: Numerical Methods (Term 1, 2022/2023) at The Chinese University of Hong Kong, Shenzhen (CUHK-SZ). The main objective of this project is to investigate and study different algorithms to compute singular value decompositions and to utilize SVDs in several interesting imaging applications. The tasks of the project can be divided into the following:

1. Implement the two-phase procedure discussed in the lecture to compute a singular value decomposition:
   $$
   \text{A} = 
    \underset{m \times n}{
        \begin{bmatrix}
            x_{11} & x_{12} & \dots & x_{1n} \\
            x_{21} & x_{22} & \dots & x_{2n} \\
            \vdots & \vdots & \ddots & \vdots \\
            x_{m1} & x_{m2} & \dots & x_{mn}
        \end{bmatrix}
    }
    = {U\Sigma V}^\text{T} = 
    \underset{m \times m}{
        \begin{bmatrix}
            | & | & \ & | \\
            u_{1} & u_{2} & \dots & u_{m} \\
            | & | & \ & |
        \end{bmatrix}
    }
    \times
    \underset{m \times n}{
    \begin{bmatrix}
    \sigma_{1} & 0 & \dots & 0 \\
    0 & \sigma_{2} & \dots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \dots & \sigma_{n} \\
    0 & 0 & \vdots & 0 \\
    \end{bmatrix}}
    \times
    \underset{n \times n}{
        \begin{bmatrix}
             & V_{1}^T &   \\
           & V_{2}^T &  \\
            \ & \vdots &  \ \\
             & V_{n}^T &  \\
        \end{bmatrix}
    }
   $$
    for any $A \in \mathbb{R}^{m \times n}$ using the Python programming language.

2. In this part, we want to explore how the SVD can be utilized in the context of deblurring image problems. This will lead to analyze which SVD algorithm is the most efficient for image compression. We will test at least three different scenarios/images and compare the performance and runtime of each SVD approaches and which algorithm output the more accuracte singular values and which one reaches convergence faster.
3. In this part of the project, we want to utilize the SVD and power iterations in order to extract the background information of some given video data and compare the performance and runtime of our implemented algorithm to standard software packages.
4. We will create a web application using the Streamlit framework providing image compression and video background extraction using our findings in the previous steps.

The complete project description can be found [here](./reports/project_description.pdf).


## Team Members

| Student ID | Student Name   | Email                        
| :----------: | :--------------: | :------------------------------: | 
| 120090727  | Darren Boesono        | 120040022@link.cuhk.edu.cn     | 
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
    │   ├── SVD                       <- Source for the main 2-Phase SVD algorithms
    │   │   │
    │   │   ├── __init__.py           <- Makes SVD a Python module
    │   │   └── two_phase_svd.py      <- The main 2-Phase SVD algorithms
    │   │
    │   └── VideoBackgroundExtractor   <- Source code for the video background extraction feature of the app
    │       │                 
    │       ├── __init__.py            <- Makes VideoBackgroundExtractor a Python Module
    │       └── extractor.py           <- Main code for the Video Background Extractor
    │   
    │
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── app.py             <- Main streamlit web application



## Installation
1. Clone this repository using the following command:
   ```
   git clone https://github.com/dboesono/SVD-ImageCompressor-VideoExtractor.git
   ```
2. Navigate to the project directory using the command:
   ```
   cd SVD-ImageCompressor-VideoExtractor
   ```
3. Create a virtual environment using the command:
   ```
   python -m venv env
   ```
4. Activate the virtual environment according to your operating system:
   - On Windows:
        ```
        .\env\Scripts\activate
        ``` 
    - On Unix or MacOS:
        ```
        source env/bin/activate
        ``` 
5. Install the necessary packages:
   ```
   pip install -r requirements.txt
   ```


## Usage
Run the following command to start the Streamlit web app:
```
streamlit run app.py
```