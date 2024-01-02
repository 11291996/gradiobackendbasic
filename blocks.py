#gathering data from users
#tictactoe example
import gradio as gr

with gr.Blocks() as demo:
    turn = gr.Textbox("X", interactive=False, label="Turn")
    board = gr.Dataframe(value=[["", "", ""]] * 3, interactive=False, type="array") #creates a dataframe of many kinds

    def place(board, turn, evt: gr.SelectData):
        if evt.value:
            return board, turn
        board[evt.index[0]][evt.index[1]] = turn
        turn = "O" if turn == "X" else "X"
        return board, turn
    #if function has a SelectData parameter, it can store the function data then use it later
    board.select(place, [board, turn], [board, turn], show_progress="hidden") #select function is event function of Dataframe
    #show_progress="hidden" hides the progress bar

demo.launch()

#multiple triggers 
with gr.Blocks() as demo1:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    trigger = gr.Textbox(label="Trigger Box")
    trigger2 = gr.Textbox(label="Trigger Box")

    def greet(name, evt_data: gr.EventData): #gr.EventData also stores the event data
        return "Hello " + name + "!", evt_data.target.__class__.__name__
    
    def clear_name(evt_data: gr.EventData):
        return "", evt_data.target.__class__.__name__
     
    gr.on( #submitting and clicking will both trigger greet function by gr.on
        triggers=[name.submit, greet_btn.click],
        fn=greet,
        inputs=name,
        outputs=[output, trigger],
    ).then(clear_name, outputs=[name, trigger2])

demo1.launch()
