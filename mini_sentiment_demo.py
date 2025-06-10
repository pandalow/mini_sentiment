import gradio as gr
from transformers import pipeline


def sentiment_analysis(text:str)->str:
    analysis_tools = pipeline("text-classification", model = "boltuix/bert-emotion",device='cuda')
    result = analysis_tools(text)
    
    return f"the label: {result[0]['label']} and the score: {result[0]['score']}"
    


demo = gr.Interface(fn = sentiment_analysis,
                    inputs=['text'],
                    outputs = ['text'],)

demo.launch()