import gradio as gr
from PIL import Image
import numpy as np

# Define a function to process click events

state = {"start":(-1,-1), "end":(-1,-1)}
def get_click_coordinates(evt: gr.SelectData, image: Image):
    x, y = evt.index
    new_image = np.zeros((image.size[1], image.size[0], 3))
    new_image = np.uint8(new_image)
    if state['start'][0] != -1:
        x1,y1 = state['start']
        new_image[min(y1,y): max(y1,y), min(x1,x): max(x1,x), :] = 255
        state['start'] = (-1,-1)
    else:
        state['start'] = (x,y)
    new_image = Image.fromarray(new_image)
    return new_image

# Create a Gradio interface
with gr.Blocks() as demo:
    # Define an image component
    image = gr.Image(type="pil", interactive=True)

    # Define a textbox to display image size and click coordinates
    output = gr.Image(type='pil')

    # Process click events and get click coordinates
    image.select(fn=get_click_coordinates, inputs=image, outputs=output)

# Run the Gradio app
demo.launch(server_name="0.0.0.0", server_port=10087)