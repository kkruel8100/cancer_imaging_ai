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

# Set the model's file path
file_path = Path("models/model_adam_scaled.h5")

# Load the model to a new object
adam_5 = tf.keras.models.load_model(file_path)

# Load env variables
load_dotenv()

# Add your OpenAI API key here
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"OpenAI API Key Loaded: {openai_api_key is not None}")


# Load the model and tokenizer for translation
model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)

# Set source language
tokenizer.src_lang = "en_XX"

# Constants
# Language information MBart
language_info = [
    "English (en_XX)",
    "Arabic (ar_AR)",
    "Czech (cs_CZ)",
    "German (de_DE)",
    "Spanish (es_XX)",
    "Estonian (et_EE)",
    "Finnish (fi_FI)",
    "French (fr_XX)",
    "Gujarati (gu_IN)",
    "Hindi (hi_IN)",
    "Italian (it_IT)",
    "Japanese (ja_XX)",
    "Kazakh (kk_KZ)",
    "Korean (ko_KR)",
    "Lithuanian (lt_LT)",
    "Latvian (lv_LV)",
    "Burmese (my_MM)",
    "Nepali (ne_NP)",
    "Dutch (nl_XX)",
    "Romanian (ro_RO)",
    "Russian (ru_RU)",
    "Sinhala (si_LK)",
    "Turkish (tr_TR)",
    "Vietnamese (vi_VN)",
    "Chinese (zh_CN)",
    "Afrikaans (af_ZA)",
    "Azerbaijani (az_AZ)",
    "Bengali (bn_IN)",
    "Persian (fa_IR)",
    "Hebrew (he_IL)",
    "Croatian (hr_HR)",
    "Indonesian (id_ID)",
    "Georgian (ka_GE)",
    "Khmer (km_KH)",
    "Macedonian (mk_MK)",
    "Malayalam (ml_IN)",
    "Mongolian (mn_MN)",
    "Marathi (mr_IN)",
    "Polish (pl_PL)",
    "Pashto (ps_AF)",
    "Portuguese (pt_XX)",
    "Swedish (sv_SE)",
    "Swahili (sw_KE)",
    "Tamil (ta_IN)",
    "Telugu (te_IN)",
    "Thai (th_TH)",
    "Tagalog (tl_XX)",
    "Ukrainian (uk_UA)",
    "Urdu (ur_PK)",
    "Xhosa (xh_ZA)",
    "Galician (gl_ES)",
    "Slovene (sl_SI)",
]

# Convert the information into a dictionary
language_dict = {}
for info in language_info:
    name, code = info.split(" (")
    code = code[:-1]
    language_dict[name] = code

# Get the language names for choices in the dropdown
languages = list(language_dict.keys())
first_language = languages[0]
sorted_languages = sorted(languages[1:])
sorted_languages.insert(0, first_language)

default_language = "English"

# Prediction responses
malignant_text = "Malignant. Please consult a doctor for further evaluation."
benign_text = "Benign. Please consult a doctor for further evaluation."

# Create instance
llm = ChatOpenAI(
    openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0
)


# Method to get system and human messages for ChatOpenAI - Predictions
def get_prediction_messages(prediction_text):
    # Create a HumanMessage object
    human_message = HumanMessage(content=f"skin lesion that appears {prediction_text}")

    # Get the system message
    system_message = SystemMessage(
        content="You are a medical professional chatting with a patient. You want to provide helpful information and give a preliminary assessment."
    )

    # Return the system message
    return [system_message, human_message]


# Method to get system and human messages for ChatOpenAI - Help
def get_chat_messages(chat_prompt):
    # Create a HumanMessage object
    human_message = HumanMessage(content=chat_prompt)

    # Get the system message
    system_message = SystemMessage(
        content="You are a medical professional chatting with a patient. You want to provide helpful information."
    )
    # Return the system message
    return [system_message, human_message]


# Method to predict the image
def predict_image(language, img):
    try:
        try:
            # Process the image
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"Error: {e}")
            return "There was an error processing the image. Please try again."

        # Get prediction from model
        prediction = adam_5.predict(img_array)
        text_prediction = "Malignant" if prediction[0][0] > 0.5 else "Benign"

        try:
            # Get the system and human messages
            messages = get_prediction_messages(text_prediction)

            # Get the response from ChatOpenAI
            result = llm(messages)

            # Get the text prediction
            text_prediction = (
                f"Prediction: {text_prediction} Explanation: {result.content}"
            )

        except Exception as e:
            print(f"Error: {e}")
            print(f"Prediction: {text_prediction}")
            text_prediction = (
                malignant_text if text_prediction == "Malignant" else benign_text
            )

        # Get selected language code
        selected_code = language_dict[language]

        # Check if the target and source languages are the same
        if selected_code == "en_XX":
            return (
                text_prediction,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        try:
            # Encode, generate tokens, decode the prediction
            encoded_text = tokenizer(text_prediction, return_tensors="pt")
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.lang_code_to_id[selected_code],
            )
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Return the result
            return (
                result[0],
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        except Exception as e:
            print(f"Error: {e}")
            return (
                f"""There was an error processing the translation. 
            In English:
            {text_prediction}
            """,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

    except Exception as e:
        print(f"Error: {e}")
        return (
            "There was an error processing the request. Please try again.",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


# Method for on submit
def on_submit(language, img):
    print(f"Language: {language}")
    if language is None or len(language) == 0:
        language = default_language
    if img is None:
        return (
            "No image uploaded. Please try again.",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return predict_image(language, img)


# Method for on clear
def on_clear():
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(visible=True),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
    )


# Method for on chat
def on_chat(language, chat_prompt):
    try:
        # Get the system and human messages
        messages = get_chat_messages(chat_prompt)
        # Get the response from ChatOpenAI
        result = llm(messages)
        # Get the text prediction
        chat_response = result.content

    except Exception as e:
        print(f"Error: {e}")
        return gr.update(
            value="There was an error processing your question. Please try again.",
            visible=True,
        ), gr.update(visible=False)

    # Get selected language code
    if language is None or len(language) == 0:
        language = default_language
    selected_code = language_dict[language]
    # Check if the target and source languages are the same
    if selected_code == "en_XX":
        return gr.update(value=chat_response, visible=True), gr.update(visible=False)

    try:
        # Encode, generate tokens, decode the prediction
        encoded_text = tokenizer(chat_response, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[selected_code]
        )
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Return the result
        return gr.update(value=result[0], visible=True), gr.update(visible=False)
    except Exception as e:
        print(f"Error: {e}")
        return (
            gr.update(
                value=f"""There was an error processing the translation. 
            In English:
            {chat_response}
            """,
                visible=True,
            ),
            gr.update(visible=False),
        )


# Gradio app

with gr.Blocks(theme=gr.themes.Default(primary_hue="green")) as demo:
    intro = gr.Markdown(
        """
    # Welcome to Skin Lesion Image Classifier!
    Select prediction language and upload image to start.
    """
    )
    language = gr.Dropdown(
        label="Response Language - Default English", choices=sorted_languages
    )
    img = gr.Image(image_mode="RGB", type="pil")
    output = gr.Textbox(label="Results", show_copy_button=True)
    chat_prompt = gr.Textbox(
        label="Do you have a question about the results or skin cancer?",
        placeholder="Enter your question here...",
        visible=False,
    )
    chat_response = gr.Textbox(
        label="Chat Response", visible=False, show_copy_button=True
    )
    submit_btn = gr.Button("Submit", variant="primary", visible=True)
    chat_btn = gr.Button("Submit Question", variant="primary", visible=False)
    submit_btn.click(
        fn=on_submit,
        inputs=[language, img],
        outputs=[output, submit_btn, chat_prompt, chat_btn, chat_response],
    )
    chat_btn.click(
        fn=on_chat, inputs=[language, chat_prompt], outputs=[chat_response, chat_btn]
    )
    clear_btn = gr.ClearButton(
        components=[language, img, output, chat_response], variant="stop"
    )
    clear_btn.click(
        fn=on_clear,
        outputs=[
            language,
            img,
            output,
            submit_btn,
            chat_prompt,
            chat_response,
            chat_btn,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)
