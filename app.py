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

def generate_importance_matrix(model_path, train_data_path):
    imatrix_command = f"./llama-imatrix -m ../{model_path} -f {train_data_path} -ngl 99 --output-frequency 10"

    os.chdir("llama.cpp")

    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in the current directory: {os.listdir('.')}")

    if not os.path.isfile(f"../{model_path}"):
        raise Exception(f"Model file not found: {model_path}")

    print("Running imatrix command...")
    process = subprocess.Popen(imatrix_command, shell=True)

    try:
        process.wait(timeout=60)  # added wait
    except subprocess.TimeoutExpired:
        print("Imatrix computation timed out. Sending SIGINT to allow graceful termination...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)  # grace period
        except subprocess.TimeoutExpired:
            print("Imatrix proc still didn't term. Forecfully terming process...")
            process.kill()

    os.chdir("..")

    print("Importance matrix generation completed.")

def split_upload_model(model_path, repo_id, oauth_token: gr.OAuthToken | None, split_max_tensors=256, split_max_size=None):
    if oauth_token.token is None:
        raise ValueError("You have to be logged in.")
    
    split_cmd = f"llama.cpp/llama-gguf-split --split --split-max-tensors {split_max_tensors}"
    if split_max_size:
        split_cmd += f" --split-max-size {split_max_size}"
    split_cmd += f" {model_path} {model_path.split('.')[0]}"
    
    print(f"Split command: {split_cmd}") 
    
    result = subprocess.run(split_cmd, shell=True, capture_output=True, text=True)
    print(f"Split command stdout: {result.stdout}") 
    print(f"Split command stderr: {result.stderr}") 
    
    if result.returncode != 0:
        raise Exception(f"Error splitting the model: {result.stderr}")
    print("Model split successfully!")
     
    
    sharded_model_files = [f for f in os.listdir('.') if f.startswith(model_path.split('.')[0])]
    if sharded_model_files:
        print(f"Sharded model files: {sharded_model_files}")
        api = HfApi(token=oauth_token.token)
        for file in sharded_model_files:
            file_path = os.path.join('.', file)
            print(f"Uploading file: {file_path}")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file,
                    repo_id=repo_id,
                )
            except Exception as e:
                raise Exception(f"Error uploading file {file_path}: {e}")
    else:
        raise Exception("No sharded files found.")
    
    print("Sharded model has been uploaded successfully!")

def process_model(model_id, q_method, use_imatrix, imatrix_q_method, private_repo, train_data_file, split_model, split_max_tensors, split_max_size, oauth_token: gr.OAuthToken | None):
    if oauth_token.token is None:
        raise ValueError("You must be logged in to use GGUF-my-repo")
    model_name = model_id.split('/')[-1]
    fp16 = f"{model_name}.fp16.gguf"

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

        imatrix_path = "llama.cpp/imatrix.dat"

        if use_imatrix:
            if train_data_file:
                train_data_path = train_data_file.name
            else:
                train_data_path = "groups_merged.txt" #fallback calibration dataset

            print(f"Training data file path: {train_data_path}")

            if not os.path.isfile(train_data_path):
                raise Exception(f"Training data file not found: {train_data_path}")

            generate_importance_matrix(fp16, train_data_path)
        else:
            print("Not using imatrix quantization.")
        username = whoami(oauth_token.token)["name"]
        quantized_gguf_name = f"{model_name.lower()}-{imatrix_q_method.lower()}-imat.gguf" if use_imatrix else f"{model_name.lower()}-{q_method.lower()}.gguf"
        quantized_gguf_path = quantized_gguf_name
        if use_imatrix:
            quantise_ggml = f"./llama.cpp/llama-quantize --imatrix {imatrix_path} {fp16} {quantized_gguf_path} {imatrix_q_method}"
        else:
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
            This model was converted to GGUF format from [`{model_id}`](https://huggingface.co/{model_id}) using llama.cpp via the ggml.ai's [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space.
            Refer to the [original model card](https://huggingface.co/{model_id}) for more details on the model.
            
            ## Use with llama.cpp
            Install llama.cpp through brew (works on Mac and Linux)
            
            ```bash
            brew install llama.cpp
            
            ```
            Invoke the llama.cpp server or the CLI.
            
            ### CLI:
            ```bash
            llama-cli --hf-repo {new_repo_id} --hf-file {quantized_gguf_name} -p "The meaning to life and the universe is"
            ```
            
            ### Server:
            ```bash
            llama-server --hf-repo {new_repo_id} --hf-file {quantized_gguf_name} -c 2048
            ```
            
            Note: You can also use this checkpoint directly through the [usage steps](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage) listed in the Llama.cpp repo as well.

            Step 1: Clone llama.cpp from GitHub.
            ```
            git clone https://github.com/ggerganov/llama.cpp
            ```

            Step 2: Move into the llama.cpp folder and build it with `LLAMA_CURL=1` flag along with other hardware-specific flags (for ex: LLAMA_CUDA=1 for Nvidia GPUs on Linux).
            ```
            cd llama.cpp && LLAMA_CURL=1 make
            ```

            Step 3: Run inference through the main binary.
            ```
            ./llama-cli --hf-repo {new_repo_id} --hf-file {quantized_gguf_name} -p "The meaning to life and the universe is"
            ```
            or 
            ```
            ./llama-server --hf-repo {new_repo_id} --hf-file {quantized_gguf_name} -c 2048
            ```
            """
        )
        card.save(f"README.md")

        if split_model:
            split_upload_model(quantized_gguf_path, new_repo_id, oauth_token, split_max_tensors, split_max_size)
        else:
            try:
                print(f"Uploading quantized model: {quantized_gguf_path}")
                api.upload_file(
                    path_or_fileobj=quantized_gguf_path,
                    path_in_repo=quantized_gguf_name,
                    repo_id=new_repo_id,
                )
            except Exception as e:
                raise Exception(f"Error uploading quantized model: {e}")
        
        
        imatrix_path = "llama.cpp/imatrix.dat"
        if os.path.isfile(imatrix_path):
            try:
                print(f"Uploading imatrix.dat: {imatrix_path}")
                api.upload_file(
                    path_or_fileobj=imatrix_path,
                    path_in_repo="imatrix.dat",
                    repo_id=new_repo_id,
                )
            except Exception as e:
                raise Exception(f"Error uploading imatrix.dat: {e}")

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
    gr.Markdown("You must be logged in to use GGUF-my-repo.")
    gr.LoginButton(min_width=250)

    model_id = HuggingfaceHubSearch(
        label="Hub Model ID",
        placeholder="Search for model id on Huggingface",
        search_type="model",
    )

    q_method = gr.Dropdown(
        ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        label="Quantization Method",
        info="GGML quantization type",
        value="Q4_K_M",
        filterable=False,
        visible=True
    )

    imatrix_q_method = gr.Dropdown(
        ["IQ3_M", "IQ3_XXS", "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q5_K_M", "Q5_K_S"],
        label="Imatrix Quantization Method",
        info="GGML imatrix quants type",
        value="IQ4_NL", 
        filterable=False,
        visible=False
    )

    use_imatrix = gr.Checkbox(
        value=False,
        label="Use Imatrix Quantization",
        info="Use importance matrix for quantization."
    )

    private_repo = gr.Checkbox(
        value=False,
        label="Private Repo",
        info="Create a private repo under your username."
    )

    train_data_file = gr.File(
        label="Training Data File",
        file_types=["txt"],
        visible=False
    )

    split_model = gr.Checkbox(
        value=False,
        label="Split Model",
        info="Shard the model using gguf-split."
    )

    split_max_tensors = gr.Number(
        value=256,
        label="Max Tensors per File",
        info="Maximum number of tensors per file when splitting model.",
        visible=False
    )

    split_max_size = gr.Textbox(
        label="Max File Size",
        info="Maximum file size when splitting model (--split-max-size). May leave empty to use the default.",
        visible=False
    )

    def update_visibility(use_imatrix):
        return gr.update(visible=not use_imatrix), gr.update(visible=use_imatrix), gr.update(visible=use_imatrix)
    
    use_imatrix.change(
        fn=update_visibility,
        inputs=use_imatrix,
        outputs=[q_method, imatrix_q_method, train_data_file]
    )

    iface = gr.Interface(
        fn=process_model,
        inputs=[
            model_id,
            q_method,
            use_imatrix,
            imatrix_q_method,
            private_repo,
            train_data_file,
            split_model,
            split_max_tensors,
            split_max_size,
        ],
        outputs=[
            gr.Markdown(label="output"),
            gr.Image(show_label=False),
        ],
        title="Create your own GGUF Quants, blazingly fast ⚡!",
        description="The space takes an HF repo as an input, quantizes it and creates a Public repo containing the selected quant under your HF user namespace.",
        api_name=False
    )

    def update_split_visibility(split_model):
        return gr.update(visible=split_model), gr.update(visible=split_model)

    split_model.change(
        fn=update_split_visibility,
        inputs=split_model,
        outputs=[split_max_tensors, split_max_size]
    )

def restart_space():
    HfApi().restart_space(repo_id="ggml-org/gguf-my-repo", token=HF_TOKEN, factory_reboot=True)

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=21600)
scheduler.start()

# Launch the interface
demo.queue(default_concurrency_limit=1, max_size=5).launch(debug=True, show_api=False)