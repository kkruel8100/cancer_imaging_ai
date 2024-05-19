## Skin Cancer Imaging AI Diagnostic Tool

#### Contributors

- Kruel, Kimberly
- McClure, Mistie
- Snyder, Todd

#### Program Structure

    └───root
        └───datasets
            │   test_benign.pkl
            │   test_malignant.pkl
            │   test.pkl
            │   train_benign.pkl
            │   train_malignant.pkl
            └───train.pkl
        └───models
            │   model_adam_5.h5
            │   model_adam_scaled.h5
            └───model.h5
        └───utils
            └───conda_list.txt
        │   .env
        │   .gitignore
        │   main.ipynb
        │   dataset_creation.ipynb
        │   dataset_load.ipynb
        │   dataset_read.ipynb
        │   gradio.ipynb
        │   modeling_hypertuning.ipynb
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

##### Step 4

Run select panels in "modeling_hypertuning". This file reads the pickle files consolidates the files into train and test dataframes and shuffles the data. Data is separated into X and y variables training and test variables. Grid search was completed to determine best parameters. Models were created and saved into models directory using both X and X scaled data. Validation accuracy and classificatin reporting determined that unscaled X data had a higher accuracy rate.

<b>Note: Grid search takes approximately 13 hours to run. Those panels are not necessary to run.

##### Step 5

Run "gradio-kk.ipynb" to create and launch gradio app. This file reads the saved model and creates a gradio app. The app allows the user to select their language, upload an image, & submit. The user is then give an image prediction in their selected language

#### Resources

Utils/conda_list.txt contains the conda environment that this app was processed in.

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
