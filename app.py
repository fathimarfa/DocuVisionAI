"""
app.py

Interactive Gradio demo for the fine-tuned OCR model. Useful both as a bonus
rubric point ("Create interactive demo") and as the visual for your demo video.

Usage:
    python app.py --checkpoint weights/checkpoint-best
"""
import argparse

import gradio as gr

from src.inference import OCRPredictor

predictor = None  # set in main()


def transcribe(image):
    if image is None:
        return "Upload or draw an image first.", ""
    result = predictor.predict(image)
    return result["text"], f"confidence: {result['confidence']}  |  latency: {result['latency_seconds']}s"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DocuVision AI") as demo:
        gr.Markdown("# DocuVision AI — Handwritten OCR Demo")
        gr.Markdown("Upload a handwritten text image and get an instant transcription from the fine-tuned TrOCR model.")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Input image")
            with gr.Column():
                text_output = gr.Textbox(label="Transcription")
                meta_output = gr.Textbox(label="Details")
        submit_btn = gr.Button("Transcribe", variant="primary")
        submit_btn.click(fn=transcribe, inputs=image_input, outputs=[text_output, meta_output])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weights/checkpoint-best")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    predictor = OCRPredictor(args.checkpoint)
    build_ui().launch(share=args.share)
