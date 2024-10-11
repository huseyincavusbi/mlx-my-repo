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

HF_TOKEN = os.environ.get("HF_TOKEN")

def process_model(model_id, q_method, private_repo, oauth_token: gr.OAuthToken | None):
    if oauth_token.token is None:
        raise ValueError("You must be logged in to use mlx-my-repo")
    model_name = model_id.split('/')[-1]

    try:
        api = HfApi(token=oauth_token.token)

        dl_pattern = ["*.md", "*.json", "*.model"]

        pattern = (
            "*.safetensors"
            if any(
                file.path.endswith(".safetensors")
                for file in api.list_repo_tree(
                    repo_id=model_id,
                    recursive=True,
                )
            )
            else "*.bin"
        )

        dl_pattern += pattern

        api.snapshot_download(repo_id=model_id, local_dir=model_name, local_dir_use_symlinks=False, allow_patterns=dl_pattern)
        print("Model downloaded successfully!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model directory contents: {os.listdir(model_name)}")

        conversion_script = "convert_hf_to_gguf.py"
        fp16_conversion = f"python llama.cpp/{conversion_script} {model_name} --outtype f16 --outfile {fp16}"
        result = subprocess.run(fp16_conversion, shell=True, capture_output=True)
        print(result)
        if result.returncode != 0:
            raise Exception(f"Error converting to fp16: {result.stderr}")
        print("Model converted to fp16 successfully!")
        print(f"Converted model path: {fp16}")

        username = whoami(oauth_token.token)["name"]
        quantized_gguf_name = f"{model_name.lower()}-{imatrix_q_method.lower()}-imat.gguf" if use_imatrix else f"{model_name.lower()}-{q_method.lower()}.gguf"
        quantized_gguf_path = quantized_gguf_name

        quantise_ggml = f"./llama.cpp/llama-quantize {fp16} {quantized_gguf_path} {q_method}"
        result = subprocess.run(quantise_ggml, shell=True, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"Error quantizing: {result.stderr}")
        print(f"Quantized successfully with {imatrix_q_method if use_imatrix else q_method} option!")
        print(f"Quantized model path: {quantized_gguf_path}")

        # Create empty repo
        new_repo_url = api.create_repo(repo_id=f"{username}/{model_name}-{imatrix_q_method if use_imatrix else q_method}-GGUF", exist_ok=True, private=private_repo)
        new_repo_id = new_repo_url.repo_id
        print("Repo created successfully!", new_repo_url)

        try:
            card = ModelCard.load(model_id, token=oauth_token.token)
        except:
            card = ModelCard("")
        if card.data.tags is None:
            card.data.tags = []
        card.data.tags.append("llama-cpp")
        card.data.tags.append("gguf-my-repo")
        card.data.base_model = model_id
        card.text = dedent(
            f"""
            # {new_repo_id}
            """
        )
        card.save(f"README.md")

        try:
            print(f"Uploading quantized model: {quantized_gguf_path}")
            api.upload_file(
                path_or_fileobj=quantized_gguf_path,
                path_in_repo=quantized_gguf_name,
                repo_id=new_repo_id,
            )
        except Exception as e:
            raise Exception(f"Error uploading quantized model: {e}")
        
        api.upload_file(
            path_or_fileobj=f"README.md",
            path_in_repo=f"README.md",
            repo_id=new_repo_id,
        )
        print(f"Uploaded successfully with {imatrix_q_method if use_imatrix else q_method} option!")

        return (
            f'Find your repo <a href=\'{new_repo_url}\' target="_blank" style="text-decoration:underline">here</a>',
            "llama.png",
        )
    except Exception as e:
        return (f"Error: {e}", "error.png")
    finally:
        shutil.rmtree(model_name, ignore_errors=True)
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


    private_repo = gr.Checkbox(
        value=False,
        label="Private Repo",
        info="Create a private repo under your username."
    )

    iface = gr.Interface(
        fn=process_model,
        inputs=[
            model_id,
            q_method,
            private_repo,
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