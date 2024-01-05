import gradio_client as grc
client = grc.Client("freddyaboulton/english-to-german")
client.deploy_discord(api_names=['german']) #/german will be the discord command