import spaces
import gradio as gr
from transformers import pipeline
from spacy import displacy
import torch

@spaces.GPU
def dummy(): # just a dummy
    pass
    
# load model pipeline globally
try:
    ner_pipe = pipeline(
        task="ner",
        model="cindyangelira/ner-roberta-large-bahasa-indonesia-finetuned",
        aggregation_strategy="simple",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define colors for each tag
ENTITY_COLORS = {
    "O": "#ffffff",        # White for 'O'
    "PER": "#ffadad",      # Light red for 'PERSON'
    "LOC": "#ffda83",      # Light yellow for 'LOCATION'
    "DATE_TIME": "#ffa500", # Light orange for 'DOB'
    "EMAIL": "#85e0e0",    # Light cyan for 'EMAIL'
    "GENDER": "#c3c3e0",   # Light gray for 'GENDER'
    "SSN": "#800080",      # Purple for 'ID'
    "PHONE": "#d1ff85"     # Light green for 'PHONE NUMBER'
}

def get_colors():
    return ENTITY_COLORS.copy()

def process_prediction(text, pred):
    if not text or not pred:
        return "<p>No text or predictions to process</p>"
        
    colors = get_colors()
    combined_ents = []
    current_ent = None
    
    try:
        for token in pred:
            token_label = token['entity_group']
            token_start = token['start']
            token_end = token['end']
            
            if current_ent is None or current_ent['label'] != token_label:
                if current_ent:
                    combined_ents.append(current_ent)
                current_ent = {
                    'start': token_start,
                    'end': token_end,
                    'label': token_label
                }
            else:
                current_ent['end'] = token_end
                
        if current_ent:
            combined_ents.append(current_ent)

        doc = {
            "text": text,
            "ents": combined_ents,
            "title": None
        }
        
        options = {"ents": list(colors.keys()), "colors": colors}
        html = displacy.render(doc, style="ent", manual=True, options=options)
        return html
        
    except Exception as e:
        return f"<p>Error processing predictions: {str(e)}</p>"

def ner_visualization(text):
    if not text or not text.strip():
        return "<p>Please enter some text</p>"
        
    try:
        predictions = ner_pipe(text)
        return process_prediction(text, predictions)
    except Exception as e:
        return f"<p>Error during NER processing: {str(e)}</p>"

# create Gradio interface
iface = gr.Interface(
    fn=ner_visualization,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text in Bahasa Indonesia..."
    ),
    outputs="html",
    title="NER Bahasa Indonesia",
    description="Enter text to see named entity recognition results highlighted.",
    examples=[
        ["Joko Widodo lahir di Surakarta pada tanggal 21 Juni 1961."],
        ["Email saya adalah example@email.com dan nomor HP 081234567890."]
    ]
)

if __name__ == "__main__":
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"Error launching interface: {e}")