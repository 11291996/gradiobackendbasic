import gradio as gr
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from gradiofrontendbasic.quickstart import demo1

#now the demo1 can be shared via generated link
demo1.launch(share = True)

#sharing in huggingface
#create a huggingface space 
#use "gradio delopy --title [space name] --app-file [launch file]"
#to update use deploy again or enable github actions option and push 
#dragging the gradio application folder to huggingface space creation page will work as well
#also personal gradio reposit can be connected to huggingface space
gr.Image(show_download_button=True) #this kind of component will have download buttons for the outputs
#gradio can be used as an api like in the quickstart's demo6
#/api/greet
#can be applied to python client or javascript client

#authentication 
demo1.launch(auth=("admin", "pass1234")) #adds authentication to the interface
#huggingface id authentication
#set hf_oauth: true in the README.md file
#use these buttons for log in 
def hello(profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "I don't know you."
    return f"Hello {profile.name}"

with gr.Blocks() as demo1:
    gr.LoginButton()
    gr.LogoutButton()
    gr.Textbox(hello) #gr.OAuthProfile will return the profile of the user

#accessing network requests
def echo(text, request: gr.Request): #use gr.Request to access the request
    if request:
        print("Request headers dictionary:", request.headers)
        print("IP address:", request.client.host)
        print("Query parameters:", dict(request.query_params))
    return text

io = gr.Interface(echo, "textbox", "textbox").launch()
#with fast api
from fastapi import FastAPI

CUSTOM_PATH = "/gradio"

app = FastAPI() 


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH) #graido interface is mounted to the fastapi

#security and file access
#set environment variable GRADIO_TEMP_DIR to access temporary files of gradio
#cache files
gr.Interface(cache_examples=True) #caches the examples
gr.Examples(cache_examples=True) #gr.Interface in gr.Blocks
#enabling access from others 
demo1.launch(allowed_paths=["/data"]) #allows access such as api or user requests to these paths as well
#disabling access from others
demo1.launch(blocked_paths=["/data"]) #blocks access to these paths