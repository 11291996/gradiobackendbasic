import gradio as gr
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from gradiofrontendbasic.quickstart import demo1
#now the demo1 can be shared via generated link
#demo1.launch(share = True)
#sharing in huggingface
#create a huggingface space 
#use "gradio delopy --title [space name] --app-file [launch file]"
#to update 
def greet(name):
    return "Hello " + name + "!"
demo = gr.Interface(greet, "textbox", "text")