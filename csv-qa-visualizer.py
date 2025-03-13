import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
from pydantic_ai import Agent
import asyncio



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
