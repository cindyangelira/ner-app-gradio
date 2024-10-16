import gradio as gr
from transformers import pipeline
from spacy import displacy
# import torch

# load model pipeline globally 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ner_pipe = pipeline(task = "ner", 
                    model="cindyangelira/ner-roberta-large-bahasa-indonesia-finetuned", 
                    aggregation_strategy = "simple",
                    device = 0)

# define colors for each tag
def get_colors():
    return {
        "O": "#ffffff",            # White for 'O'
        "PER": "#ffadad",      # Light red for 'PERSON'
        "LOC": "#ffda83",    # Light yellow for 'LOCATION'
        "DATE_TIME": "#ffa500",         # Light orange for 'DOB'
        "EMAIL": "#85e0e0",       # Light cyan for 'EMAIL'
        "GENDER": "#c3c3e0",      # Light gray for 'GENDER'
        "SSN": "#800080",          # Purple for 'ID'
        "PHONE": "#d1ff85" # Light green for 'PHONE NUMBER'
    }


def process_prediction(text, pred):
    colors = get_colors()
    combined_ents = []     # initialize an empty list to store combined entities

    current_ent = None # var to track current entitiy

    for token in pred:
        token_label = token['entity_group'] #.replace('B-', '').replace('I-', '')
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
        combined_ents.append(current_ent) # add the last entity after the loop finishes


    doc = {  # doc for viz
        "text": text,
        "ents": combined_ents,
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
        fn=ner_visualization,                 
        inputs=gr.Textbox(label="Input Text"),
        outputs="html",                      
        title="NER Bahasa Indonesia",           
        description="Enter text to see named entity recognition results highlighted."
    )
    return iface


if __name__ == "__main__":
    app = build_interface()
    app.launch()
