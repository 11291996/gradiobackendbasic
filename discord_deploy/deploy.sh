gradio deploy --title echo-chatbot --app-file app.py #this will deploy the app to gradio.app
#in deploying let's say that huggingface space is used, then directly it can be deployed to discord
gradio deploy-discord --src freddyaboulton/echo-chatbot
#add the bot to a discord token 
gradio deploy-discord --src freddyaboulton/echo-chatbot --discord-bot-token <token>
#go to the space and  “Add this bot to your server by clicking this link:” and do the following to add it to a discord server 
#use /chat to use the bot in discord
#do this to directly deploy to discord
gradio deploy --title echo-chatbot --app-file deploy.py
#then those gradio commands are not needed 
gradio deploy --title echo-chatbot --app-file translator.py #this will deploy non chatbot api to discord 