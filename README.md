Image Compression & Video Background Extraction using SVD
==============================


Project Organization
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


