import time
import shap
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import torch
import numpy as np
import scipy as sp
import pandas as pd


#device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering , AutoModelForSequenceClassification
from transformers import pipeline


train_df = pd.read_pickle("./train.pkl")
test_df = pd.read_pickle("./test.pkl")

tokenizer_class = AutoTokenizer.from_pretrained("Nakul24/RoBERTa-emotion-classification")

model_class = AutoModelForSequenceClassification.from_pretrained("Nakul24/RoBERTa-emotion-classification")

tokenizer_qa = AutoTokenizer.from_pretrained("Nakul24/RoBERTa-emotion-extraction")
model_qa = AutoModelForQuestionAnswering.from_pretrained("Nakul24/RoBERTa-emotion-extraction")

Classification = pipeline(task="text-classification", model=model_class, tokenizer=tokenizer_class, device = 0)
question_answering = pipeline(task="question-answering", model=model_qa, tokenizer=tokenizer_qa, device = 0)

#question = "angry"
#context = "The table was scratched by the cat now it is worthless"
#result = question_answering(question=question, context=context)
#search_value = result['answer'] # span


# Switch to cuda, eval mode, and FP16 for faster inference
if device == "cuda":
    model_qa = model_qa.half()
    model_class = model_class.half()
model_qa.to(device)
model_qa.eval()
model_class.to(device)
model_class.eval()

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
server = app.server

min_ind = 0
max_ind = 800

def iou(ins,outs):
  s1 = ins
  s2 = outs
  u = s1.union(s2)
  inter = s1.intersection(s2)
  iou = len(inter)/len(u)
  return iou


controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Choose the index value"),
                dbc.Input(id="index-num",type="number", min=min_ind, max=max_ind, step=1,value=0)                
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Input Data Source"),
                dbc.RadioItems(
                    options=[
                        {"label": "Train", "value": 1},
                        {"label": "Test", "value": 2},
                        {"label": "Custom", "value": 3},
                    ],
                    value=3,
                    id="radioitems-inline-input",
                    inline=True,
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Spinner(
                    [
                        dbc.Button("Explain!", id="button-run"),
                        html.Div(id="time-taken"),
                    ]
                )
            ]
        ),
    ],
    body=True,
    style={"height": "275px"},
)


# Define Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Dash Emotion Extraction (with RoBERTa)"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    width=6,
                    children=[
                        controls,
                        dbc.Card(
                            body=True,
                            children=[
                                dbc.FormGroup(
                                    [
                                        dbc.Label("RoBERTa-Cause-Explainer"),
                                        html.Br(),
                                        html.Br(),
                                        html.Div(
                                        id="summarized-content"
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Br(),
                                        html.Div(
                                        id="iou_r1"
                                        ),
                                            
                                        
                                    ]
                                )
                            ],
                        ),
                    ],
                ),
                dbc.Col(
                    width=6,
                    children=[
                        dbc.Card(
                            body=True,
                            children=[
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Context (Paste here)"),
                                        dcc.Textarea(
                                            id="original-text",
                                            style={"width": "100%", "height": "200px"},
                                        ),
                                    ]
                                )
                            ],
                        ),
                        dbc.Card(
                            body=True,
                            children=[
                                dbc.FormGroup(
                                    [
                                        dbc.Label("SHAP-Cause-Explainer"),
                                        html.Br(),
                                        html.Br(),
                                        html.Div(
                                        id="summarized-content-SHAP"
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Br(),
                                       
                                        html.Div(
                                        id="iou_s1"
                                        ),
                                    ]
                                )
                            ],
                        )
                    ],
                ),
            ]
        ),
    ],
)


'''
@app.callback(
    [Output("index-num", "max")],
    [
        Input("button-run", "n_clicks"),
        Input("radioitems-inline-input", "value"),
    ],
)
def set_min_max(n_clicks,value):
    if value == 1:
        return (len(train_df))
    if value == 2:
        return (len(test_df))
    else: return ('100')
'''



@app.callback(
    Output("original-text", "value"),
    [   Input("button-run", "n_clicks"),
        Input("radioitems-inline-input", "value"),
        Input("index-num", "value"),
    ],
)
def read_from_df(n_clicks,radio,index):
    if radio == 1:
        return (train_df['original_situation'][index])
    if radio == 2:
        return (test_df['original_situation'][index])
    else: return ""




@app.callback(
    [Output("summarized-content", "children"), Output("time-taken", "children"),Output("summarized-content-SHAP", "children"),Output("iou_r1", "children"),Output("iou_s1", "children")],
    [
    Input("button-run", "n_clicks"),
    ],    
        
    
    [State("index-num", "value"),State("radioitems-inline-input", "value"),State("original-text", "value")],
)
def summarize(n_clicks, index, radio, original_text):
    if original_text is None or original_text == "":
        return "", "NO OUTPUT","","",""

    t0 = time.time()
    emotion = Classification(original_text)
    result = question_answering(question=emotion[0]['label'], context=original_text)
    text = original_text

    output = html.Div([text])

    sequences = text.split(result['answer'])
    i = 1
    while i < len(sequences):
        sequences.insert(i, html.Mark(result['answer'], style={
            "background": 'cyan',
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "line-height": "1",
            "border-radius": "0.35em",
        }))
        i += 2

    output.children = sequences

    labels = [x[0] for x in sorted(model_class.config.label2id.items(), key=lambda x: x[1])]

    # define a prediction function
    def f(x):
        tv = torch.tensor([tokenizer_class.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
        attention_mask = (tv!=0).type(torch.int64).cuda()
        outputs = model_class(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:,1]) # use one vs rest logit units
        
        return val

    explainer = shap.Explainer(f, tokenizer_class)
    shap_values = explainer([text], fixed_context=1)
    s1 = []
    for i in range (0, len(shap_values[0]) - 1):
        if (shap_values[0][i].values > 0 and shap_values[0][i].data != ''):
            s1.append(shap_values[0][i].data)
    
    
    output2 = html.Div([text])

    sequences1 = []
    s2 = list(map(str.strip, s1))
    for x in text.split():
        if(x in s2):
            sequences1.append(html.Mark(x + " ", style={
                "background": 'cyan',
                "padding": "0.45em 0.6em",
                "margin": "0 0.25em",
                "line-height": "1",
                "border-radius": "0.35em",
            }))
        else :
            sequences1.append(x + " ")
    output2.children = sequences1

    #s2 is the list of words
    # label is human list of words
    # result['answer'] is the string

    s_rob = set(result['answer'].split(' '))
    label_h = []
    if radio == 1:
        label_h = train_df['labels'][index]
    elif radio == 2:
        label_h = test_df['labels'][index]
    else: label_h = [""]
    s_hum = set(label_h)
    s_shap = set(s2)
    iou_r = float(iou(s_hum,s_rob))
    iou_s = float(iou(s_hum,s_shap))
    
    output3 = html.Div([text])
    seq = []
    seq.append(html.P(str("Human annotations are: " + str(label_h))))
    seq.append('\n')
    seq.append(html.P(str("IOU with human annotation is: " + str(iou_r))))
    output3.children = seq
    
    output4 = html.Div([text])
    seq = []
    seq.append(html.P(str("Human annotations are: " + str(label_h))))
    seq.append('\n')
    seq.append(html.P(str("IOU with human annotation is: " + str(iou_s))))
    #seq.append(str("IOU with human annotation is: " + str(iou_s)))
    output4.children = seq
    #sequences.append(str("\nIOU with human annotation is: " + str(iou_r)))
    #output.children = sequences
    
    t1 = time.time()
    time_taken = f"Summarized on {device} in {t1-t0:.2f}s. Predicted Emotion is {emotion[0]['label']}"

    return output , time_taken , output2, output3 ,output4 


if __name__ == "__main__":
    app.run_server(debug=True)
