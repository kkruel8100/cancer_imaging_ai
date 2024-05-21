# Skin Cancer Imaging AI Diagnostic Tool

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

Images were downloaded from Kaggle to the local machine. The `dataset_creation.ipynb` file was run to create pickle files from the 4 datasets: test/benign, test/malignant, train/benign, and train/malignant. The pickle files were saved to `/datasets`. Due to size limitations, the pickle files were excluded from GitHub.

**Note:** This step does not need to be run to recreate/run other files.

##### Step 2:

Pickle files were loaded to AWS S3 storage. The `dataset_load.ipynb` file loads the pickle files from AWS and saves them locally.

**Note:** `.env` file is required to access AWS. To save costs, the files are saved into the local folder for developer use. This file only needs to be run once to create local files.

##### Step 3:

Pickle files are read from local. The `dataset_read.ipynb` reads the pickle files from local directory files that were created in Step 2.

##### Step 4:

Run select panels in `modeling_hypertuning.ipynb`. This file reads the pickle files, consolidates the files into train and test dataframes, and shuffles the data. Data is separated into X and y variables, training and test variables. Grid search was completed to determine best parameters. Models were created and saved to `/models`.

**Note:** Only select panels need to be run to recreate models.

##### Step 5

Run "gradio-kk.ipynb" to create and launch gradio app. This file reads the saved model and creates a gradio app. The app allows the user to select their language, upload an image, & submit. The user is then give an image prediction in their selected language

# Gradio Interface

The Gradio interface provides an easy-to-use web interface for the Skin Cancer Imaging AI Diagnostic Tool. This allows users to upload an image of a skin lesion and receive a diagnostic prediction.

### The following steps outline the process:

##### 1. Import Libraries:
import pickle
from PIL import Image
import numpy as np
import gradio as gr
from pathlib import Path
from transformers import pipeline
from tensorflow.keras.models import load_model
import tensorflow as tf
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from dotenv import load_dotenv
import openai
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

#### 2. Load Model
file_path = Path("models/model_adam_5.h5")
adam_5 = tf.keras.models.load_model(file_path)

#### 3. Setup Gradio Interface:
def preprocess_image(image):
    """
    Preprocess the input image to the required format for prediction.
    Args:
        image (PIL.Image): The input image uploaded by the user.
    Returns:
        np.array: The preprocessed image as a NumPy array.
    """
    # Resize the image to match the model's expected input size
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array and scale pixel values
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to match the model's expected input shape
    return np.expand_dims(image_array, axis=0)

def predict(image):
    """
    Make a prediction on the input image using the loaded model.
    Args:
        image (PIL.Image): The input image uploaded by the user.
    Returns:
        str: The prediction label (e.g., 'Benign' or 'Malignant').
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Use the model to make a prediction
    prediction = adam_5.predict(processed_image)
    
    # Convert the prediction to a label
    if prediction[0][0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

# Create and launch the Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Label(),
    title="Skin Cancer Diagnostic Tool",
    description="Upload an image of a skin lesion to receive a diagnostic prediction."
).launch()


This setup allows users to upload skin images and receive diagnostic predictions through an easy-to-use web interface. The model used for prediction is model_adam_5.h5, and it leverages TensorFlow and Gradio for deployment

To launch the Gradio interface, run the `gradio.ipynb` notebook. The main components of the interface are:

- **Language Selection**: Users can select the prediction language from a dropdown menu. The default language is English.
- **Image Upload**: Users can upload an image of a skin lesion.
- **Submit Button**: Users can submit the image for prediction.
- **Prediction Output**: The diagnostic prediction is displayed in a textbox.
- **Questions/ Chat Input**: Open field to typr out questions.
- **Submit Question**: Users can submit questions for answers from a chatbot.
- **Answers to Questions Output**: Answers to questions are provided from the chatbot.
- **Clear Button**: Users can clear the inputs and outputs.

##### Application Function

1. **Launch Gradio Interface**:
    ```bash
    jupyter notebook gradio.ipynb
    ```
2. **Follow the Instructions**:
    - Select the prediction language.
    - Upload an image of the skin lesion.
    - Click the submit button to receive the diagnostic prediction.
    - Use the clear button to reset the interface for a new prediction.

#### Screenshot
The interface will look something like this:

![Screenshot](./images/gradio_screenshot_1.png)
![Screenshot](./images/gradio_screenshot_2.png)
##### Resources

Utils/conda_list.txt contains the conda environment that this app was processed in.

#### Extrernal Link
Hugging Face
https://huggingface.co/spaces/kkruel/skin_cancer_detection_ai

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

