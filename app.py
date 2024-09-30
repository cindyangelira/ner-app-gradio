import gradio as gr
from transformers import pipeline
from spacy import displacy

# load model pipeline globally 
ner_pipe = pipeline("token-classification", model="cindyangelira/ner-roberta-large-bahasa-indonesia")

# define colors for each tag
def get_colors():
    return {
        "O": "#ffffff",            # White for 'O'
        "PERSON": "#ffadad",      # Light red for 'PERSON'
        "LOCATION": "#ffda83",    # Light yellow for 'LOCATION'
        "DOB": "#ffadad",         # Light red for 'DOB'
        "EMAIL": "#85e0e0",       # Light cyan for 'EMAIL'
        "GENDER": "#c3c3e0",      # Light gray for 'GENDER'
        "ACCOUNT": "#b0e0e6",     # Light blue for 'ACCOUNT'
        "ID": "#800080",          # Purple for 'ID'
        "PHONE": "#d1ff85" # Light green for 'PHONE NUMBER'
    }


def process_prediction(text, pred):
    colors = get_colors()

    for token in pred:
        token['label'] = token['entity'].replace('B-', '').replace('I-', '')

    ents = [{'start': token['start'], 'end': token['end'], 'label': token['label']} for token in pred]

    doc = {
        "text": text,
        "ents": ents,
        "title": None
    }

    options = {"ents": list(colors.keys()), "colors": colors}
    html = displacy.render(doc, style="ent", manual=True, options=options)
    return html


def ner_visualization(text):
    predictions = ner_pipe(text)  
    return process_prediction(text, predictions)


def build_interface():
    iface = gr.Interface(
        fn=ner_visualization,                 # Main function for NER visualization
        inputs=gr.Textbox(label="Input Text"),# Input textbox
        outputs="html",                       # Output is HTML with rendered NER
        title="NER Bahasa Indonesia",            # Title of the app
        description="Enter text to see named entity recognition results highlighted."
    )
    return iface


if __name__ == "__main__":
    app = build_interface()
    app.launch()
