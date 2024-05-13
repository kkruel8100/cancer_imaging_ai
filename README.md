## Skin Cancer Imaging AI Diagnostic Tool

#### Contributors

- Kruel, Kimberly
- McClure, Mistie
- Snyder, Todd

#### Program Structure

    └───root
        └───datasets
            │   test_benign.pkl
            |   test_malignant.pkl
            │   train_benign.pkl
            └───train_malignant.pkl
        └───utils
            │
            │
            └───
        │   .env
        │   .gitignore
        │   main.ipynb
        |   dataset_creation.ipynb
        |   dataset_load.ipynb
        └───README.md

#### Data Processing

##### Step 1:

Images were downloaded from Kaggle to local machine. "dataset_creation.ipynb" file was ran to create pickle files from the 4 datasets: test/benign, test/malignant, train/benign, and test_malignant. The pickle files were saved to /datasets. Due to size limitations, the pickle files were excluded from Github.

<b>Note: This step does not need to be ran to recreate/run other files.

##### Step 2:

Pickle files were loaded to AWS S3 storage. "dataset_load.ipynb" file loads the pickle files from AWS. It then saves the files into local.

<b>Note: .env file is required to access AWS. To save costs, the files are saved into local folder for developer use. This file only needs to be ran once to create local files.<b>

##### Step 3:

Pickle files are read from local. "dataset_read.ipynb" reads the pickle files from local directory files that were created in Step 2.

#### Resources

##### Datasets

Melanoma Cancer Image Dataset
https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset

License
[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

Sources
"The melanoma cancer dataset was sourced through a comprehensive approach to ensure diversity and representativeness."

Collection Methodology
"The images were collected from publicly available resources."

#### Notice

The AI tool has been developed for entertainment purposes and is not intended to provide medical advice.
