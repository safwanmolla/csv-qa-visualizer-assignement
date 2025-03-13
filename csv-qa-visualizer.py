import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
from pydantic_ai import Agent
import asyncio


class CSVQA(Agent):
    model: str = "llama3"

    async def run(self, question: str, csv_data: str):

        self.system_prompt = f"Use the following data to answer the user's question:\n\n{csv_data}"
        return await super().run(question)


def process_question(question, csv_file):
    try:
        df = pd.read_csv(csv_file.name)
        data_summary = df.describe(include='all').to_string()
        # response = ollama.chat(model='llama3.1-8b', messages=[{'role': 'user', 'content': f'Question: {question}\\nData:\\n{data_summary}'}])
        # response = response.get('content', 'No response from LLM.')
        # return response

        csv_qa_agent = CSVQA(model="llama3")
        response = asyncio.run(csv_qa_agent.run(question, data_summary))
        return response if response else "No response from LLM."

    except Exception as e:
        return f"Error processing file: {str(e)}"


def generate_plot(column_name, csv_file):
    try:
        df = pd.read_csv(csv_file.name)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in CSV."

        plt.figure(figsize=(6, 4))
        df[column_name].hist(bins=30, edgecolor='black')
        plt.title(f"Distribution of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")

        plot_path = "plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        return f"Error generating plot: {str(e)}"

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
