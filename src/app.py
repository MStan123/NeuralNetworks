import gradio as gr
from predict import predict

interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(),
    outputs=gr.Textbox(label="Prediction Result"),
    title='Handwritten Digit Recognition',
    description='Draw a digit (0-9) and submit!'
)

if __name__ == "__main__":
    interface.launch()
