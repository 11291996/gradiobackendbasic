import gradio as gr

#global state for the interface
#saves data for entire launch
scores = [] #this is outside of the interface, so global, which means every user of this launch will have access to this data

def track_score(score):
    scores.append(score)
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores

demo1 = gr.Interface(
    track_score, 
    gr.Number(label="Score"), 
    gr.JSON(label="Top Scores") #displays json format clearer 
)

demo1.launch()

#session state for the interface
#save data for each users
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def user(message, history):
    return "", history + [[message, None]]


def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(
        user_message + tokenizer.eos_token, return_tensors="pt"
    )

    #append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor([]), new_user_input_ids], dim=-1)

    #generate a response
    response = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    ).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(response[0]).split("<|endoftext|>")
    response = [
        (response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)
    ]  # convert to tuples of list
    history[-1] = response[0]
    return history


with gr.Blocks() as demo2:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    ) #submit uses the function when it is clicked, gets the defined input and returns the defined output
    #then uses the function continually when it is clicked after the submit button is clicked
    #queue is the same with interface queue but for gradio objects
    #submit creates sessions for each user 
    #then saves the data for each user
    #this is a model for session state
    clear.click(lambda: None, None, chatbot, queue=False)

demo2.launch()

#loading it from database outside 
gr.Interface.load("huggingface/gpt2").launch();
#customizes the input component
gr.Interface.load("huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text")  
).launch()
#loading from already shared huggingface space
gr.Interface.load("spaces/eugenesiow/remove-bg",
                  inputs="webcam",
                  title="Remove your webcam background!").launch()
#returing output
io = gr.Interface.load("models/EleutherAI/gpt-neo-2.7B")
io("It was the best of times")

#kinds of interfaces
#standard demos have input and output defined 
#output only demo 
import time

def fake_gan():
    time.sleep(1)
    images = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
            "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
            "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
    ]
    return images


demo3 = gr.Interface(
    fn=fake_gan,
    inputs=None,
    outputs=gr.Gallery(label="Generated Images", columns=[2]),
    title="FD-GAN",
    description="This is a fake demo of a GAN. In reality, the images are randomly chosen from Unsplash.",
)

demo3.launch() #each session just creates an image, no interaction 

#input only demo 
import random
import string

def save_image_random_name(image):
    random_string = ''.join(random.choices(string.ascii_letters, k=20)) + '.png'
    image.save(random_string)
    print(f"Saved image to {random_string}!")

demo4 = gr.Interface(
    fn=save_image_random_name, 
    inputs=gr.Image(type="pil"), 
    outputs=None,
)
demo4.launch() #just saves the data in database and no output returned

#unified demos
generator = pipeline('text-generation', model = 'gpt2')

def generate_text(text_prompt):
  response = generator(text_prompt, max_length = 30, num_return_sequences=5)
  return response[0]['generated_text']

textbox = gr.Textbox()

demo5 = gr.Interface(generate_text, textbox, textbox)

demo5.launch() #output and input are channeled in the same component