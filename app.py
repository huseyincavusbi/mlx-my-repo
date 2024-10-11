import os
import shutil
import subprocess
import signal
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr

from huggingface_hub import create_repo, HfApi
from huggingface_hub import snapshot_download
from huggingface_hub import whoami
from huggingface_hub import ModelCard

from gradio_huggingfacehub_search import HuggingfaceHubSearch

from apscheduler.schedulers.background import BackgroundScheduler

from textwrap import dedent

from mlx_lm import convert

HF_TOKEN = os.environ.get("HF_TOKEN")

def process_model(model_id, q_method,oauth_token: gr.OAuthToken | None):
    if oauth_token.token is None:
        raise ValueError("You must be logged in to use GGUF-my-repo")
    model_name = model_id.split('/')[-1]
    username = whoami(oauth_token.token)["name"]
    
    try:
        upload_repo = username + "/" + model_name + "-mlx"
        convert(model_id, quantize=True, upload_repo=upload_repo)
        return (
            f'Find your repo <a href=\'{new_repo_url}\' target="_blank" style="text-decoration:underline">here</a>',
            "llama.png",
        )
    except Exception as e:
        return (f"Error: {e}", "error.png")
    finally:
        shutil.rmtree("mlx_model", ignore_errors=True)
        print("Folder cleaned up successfully!")

css="""/* Custom CSS to allow scrolling */
.gradio-container {overflow-y: auto;}
"""
# Create Gradio interface
with gr.Blocks(css=css) as demo: 
    gr.Markdown("You must be logged in to use MLX-my-repo.")
    gr.LoginButton(min_width=250)

    model_id = HuggingfaceHubSearch(
        label="Hub Model ID",
        placeholder="Search for model id on Huggingface",
        search_type="model",
    )

    q_method = gr.Dropdown(
        ["Q4", "Q8"],
        label="Quantization Method",
        info="MLX quantization type",
        value="Q4",
        filterable=False,
        visible=True
    )
    
    iface = gr.Interface(
        fn=process_model,
        inputs=[
            model_id,
            q_method,
        ],
        outputs=[
            gr.Markdown(label="output"),
            gr.Image(show_label=False),
        ],
        title="Create your own MLX Quants, blazingly fast ⚡!",
        description="The space takes an HF repo as an input, quantizes it and creates a Public/ Private repo containing the selected quant under your HF user namespace.",
        api_name=False
    )

def restart_space():
    HfApi().restart_space(repo_id="reach-vb/mlx-my-repo", token=HF_TOKEN, factory_reboot=True)

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=21600)
scheduler.start()

# Launch the interface
demo.queue(default_concurrency_limit=1, max_size=5).launch(debug=True, show_api=False)