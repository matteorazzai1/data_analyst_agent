#!/usr/bin/env python
"""
Gradio interface for the DataAnalystAgent.
Provides a web-based chatbot interface with example questions.
"""

import gradio as gr
import sys
from pathlib import Path

# Add the project root to the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent import DataAnalystAgent, EXAMPLES

# Global agent instance (initialized once)
agent_instance = None


def initialize_agent():
    """Initialize the agent (lazy loading)."""
    global agent_instance
    if agent_instance is None:
        print("[Gradio] Initializing DataAnalystAgent...")
        agent_instance = DataAnalystAgent()
    return agent_instance


def chat(question, chat_history):

    if chat_history is None:
        chat_history = []

    answer = str(initialize_agent().answer(question))

    chat_history.append(
        {"role": "user", "content": question}
    )

    chat_history.append(
        {"role": "assistant", "content": answer}
    )

    return chat_history, ""


# Build the Gradio interface
with gr.Blocks(title="DataAnalystAgent - WDI Chatbot") as demo:

    gr.Markdown(
        """
# DataAnalystAgent — World Development Indicator Chatbot

A conversational AI agent that answers questions about World Development Indicators (WDI) by combining:

- **Semantic Retrieval (RAG)**: Searches indicator documentation
- **Code Execution**: Analyzes data using Python

Ask any question about WDI data or indicators!
"""
    )

    chatbot = gr.Chatbot(
        label="Chat History",
        height=400,
    )

    question_input = gr.Textbox(
        label="Ask a question",
        placeholder="Type your question here...",
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear History")

    gr.Markdown("### Example Questions")
    gr.Markdown("Click any example below to load it into the question field:")

    with gr.Row():
        for i in range(0, len(EXAMPLES), 2):

            with gr.Column():

                if i < len(EXAMPLES):
                    example_text = EXAMPLES[i]

                    gr.Button(
                        value=f"📌 {example_text[:50]}...",
                        size="sm",
                    ).click(
                        fn=lambda text=example_text: text,
                        outputs=question_input,
                    )

                if i + 1 < len(EXAMPLES):
                    example_text = EXAMPLES[i + 1]

                    gr.Button(
                        value=f"📌 {example_text[:50]}...",
                        size="sm",
                    ).click(
                        fn=lambda text=example_text: text,
                        outputs=question_input,
                    )

    with gr.Accordion("About this agent", open=False):

        gr.Markdown(
            """
### Architecture
- **Orchestration**: LlamaIndex ReActAgent with 4 specialized tools
- **Retrieval**: sentence-transformers + FAISS semantic search
- **Data**: World Development Indicators (WDI) CSV dataset
- **Execution**: Restricted Python sandbox for safe code execution

### Tools
1. **inspect_schema**: Look up indicator codes and explore data structure
2. **retrieve_docs**: Search indicator documentation semantically
3. **retrieve_metadata**: Find data quality notes and country info
4. **run_python**: Execute pandas/numpy code for analysis

### Dataset
- 190+ countries and regions
- 1,500+ economic, social, and environmental indicators
- Historical data from 1960 to present
"""
        )

    submit_btn.click(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )

    question_input.submit(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, question_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )