import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
from pydantic_ai import Agent
import asyncio


def process_question(question, csv_file):
    try:
        df = pd.read_csv(csv_file.name)
        data_summary = df.describe(include='all').to_string()
        response = ollama.chat(model='llama3.1-8b', messages=[{'role': 'user', 'content': f'Question: {question}\\nData:\\n{data_summary}'}])
        response = response.get('content', 'No response from LLM.')
        return response
    except Exception as e:
        return f"Error processing file: {str(e)}"

def load_sample_csv():
    try:
        df = pd.read_csv("MELBOURNE_HOUSE_PRICES.csv")
        return df.head().to_string()
    except Exception as e:
        return f"Error loading sample CSV: {str(e)}"


with gr.Blocks() as app:
    gr.Markdown("# CSV Question Answering & Visualization")
    file_input = gr.File(label="Upload CSV File")
    question_input = gr.Textbox(label="Enter your question")
    answer_output = gr.Textbox(label="Answer")
    ask_button = gr.Button("Ask")

    column_input = gr.Textbox(label="Enter column name for visualization")
    plot_output = gr.Image(label="Generated Plot")
    plot_button = gr.Button("Generate Plot")

    sample_data_output = gr.Textbox(label="Sample Data", interactive=False)
    load_sample_button = gr.Button("Load Sample CSV")

    ask_button.click(process_question, inputs=[question_input, file_input], outputs=answer_output)
    plot_button.click(generate_plot, inputs=[column_input, file_input], outputs=plot_output)
    load_sample_button.click(load_sample_csv, outputs=sample_data_output)

app.launch()
