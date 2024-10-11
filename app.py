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
from huggingface_hub import login
from huggingface_hub import scan_cache_dir
from huggingface_hub import logging

from gradio_huggingfacehub_search import HuggingfaceHubSearch

from apscheduler.schedulers.background import BackgroundScheduler

from textwrap import dedent

from mlx_lm import convert, __version__

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

HF_TOKEN = os.environ.get("HF_TOKEN")

def clear_cache():
    scan = scan_cache_dir()
    to_delete = []
    for repo in scan.repos:
        if repo.repo_type == "model":
            to_delete.append([rev.commit_hash for rev in repo.revisions])

    scan.delete_revisions(*to_delete)

    print("Cache has been cleared")

def upload_to_hub(path, upload_repo, hf_path, token):
    
    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.data.base_model = hf_path
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using mlx-lm version **{__version__}**.

        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")

        prompt="hello"

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{{"role": "user", "content": prompt}}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")    

def process_model(model_id, q_method,oauth_token: gr.OAuthToken | None):
    
    if oauth_token.token is None:
        raise ValueError("You must be logged in to use MLX-my-repo")
    
    model_name = model_id.split('/')[-1]
    username = whoami(oauth_token.token)["name"]

    # login(token=oauth_token.token, add_to_git_credential=True)
    
    try:
        upload_repo = username + "/" + model_name + "-mlx"
        convert(model_id, quantize=True)
        upload_repo(path="mlx_model", upload_repo=upload_repo, hf_path=repo_id, token=oauth_token.token)
        clear_cache()
        return (
            f'Find your repo <a href=\'{new_repo_url}\' target="_blank" style="text-decoration:underline">here</a>',
            "llama.png",
        )
    except Exception as e:
        return (f"Error: {e}", "error.png")
    finally:
        shutil.rmtree("mlx_model", ignore_errors=True)
        clear_cache()
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