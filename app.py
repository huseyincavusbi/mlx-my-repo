import os
import shutil
import subprocess

import gradio as gr

from huggingface_hub import create_repo, HfApi
from huggingface_hub import snapshot_download
from huggingface_hub import whoami
from huggingface_hub import ModelCard

from textwrap import dedent

LLAMA_LIKE_ARCHS = ["MistralForCausalLM", "LlamaForCausalLM"]

def script_to_use(model_id, api):
    info = api.model_info(model_id)
    if info.config is None:
        return None
    arch = info.config.get("architectures", None)
    if arch is None:
        return None
    arch = arch[0]
    return "convert.py" if arch in LLAMA_LIKE_ARCHS else "convert-hf-to-gguf.py"

def process_model(model_id, q_method, hf_token):
    model_name = model_id.split('/')[-1]
    fp16 = f"{model_name}/{model_name.lower()}.fp16.bin"
    
    try:
        api = HfApi(token=hf_token)

        snapshot_download(repo_id=model_id, local_dir=model_name, local_dir_use_symlinks=False)
        print("Model downloaded successully!")
        
        conversion_script = script_to_use(model_id, api)
        fp16_conversion = f"python llama.cpp/{conversion_script} {model_name} --outtype f16 --outfile {fp16}"
        result = subprocess.run(fp16_conversion, shell=True, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"Error converting to fp16: {result.stderr}")
        print("Model converted to fp16 successully!")

        qtype = f"{model_name}/{model_name.lower()}.{q_method.upper()}.gguf"
        quantise_ggml = f"./llama.cpp/quantize {fp16} {qtype} {q_method}"
        result = subprocess.run(quantise_ggml, shell=True, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"Error quantizing: {result.stderr}")
        print("Quantised successfully!")

        # Create empty repo
        new_repo_url = api.create_repo(repo_id=f"{model_name}-{q_method}-GGUF", exist_ok=True)
        new_repo_id = new_repo_url.repo_id
        print("Repo created successfully!", new_repo_url)

        card = ModelCard.load(model_id)
        card.data.tags = ["llama-cpp"] if card.data.tags is None else card.data.tags + ["llama-cpp"]
        card.text = dedent(
            f"""
            # {new_repo_id}
            This model was converted to GGUF format from [`{model_id}`](https://huggingface.co/{model_id}) using llama.cpp.
            Refer to the [original model card](https://huggingface.co/{model_id}) for more details on the model.
            ## Use with llama.cpp

            ```bash
            brew install ggerganov/ggerganov/llama.cpp
            ```

            ```bash
            llama-cli --hf-repo {new_repo_id} --model {qtype.split("/")[-1]} -p "The meaning to life and the universe is "
            ```

            ```bash
            llama-server --hf-repo {new_repo_id} --model {qtype.split("/")[-1]} -c 2048
            ```
            """
        )
        card.save(os.path.join(model_name, "README-new.md"))

        api.upload_file(
            path_or_fileobj=qtype,
            path_in_repo=qtype.split("/")[-1],
            repo_id=new_repo_id,
        )

        api.upload_file(
            path_or_fileobj=f"{model_name}/README-new.md",
            path_in_repo="README.md",
            repo_id=new_repo_id,
        )
        print("Uploaded successfully!")

        return (
            f'Find your repo <a href=\'{new_repo_url}\' target="_blank" style="text-decoration:underline">here</a>',
            "llama.png",
        )
    except Exception as e:
        return (f"Error: {e}", "error.png")
    finally:
        shutil.rmtree(model_name, ignore_errors=True)
        print("Folder cleaned up successfully!")


# Create Gradio interface
iface = gr.Interface(
    fn=process_model, 
    inputs=[
        gr.Textbox(
            lines=1, 
            label="Hub Model ID",
            info="Model repo ID",
        ),
        gr.Dropdown(
            ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"], 
            label="Quantization Method", 
            info="GGML quantisation type",
            value="Q4_K_M",
            filterable=False
        ),
        gr.Textbox(
            lines=1, 
            label="HF Write Token",
            info="https://hf.co/settings/token",
            type="password",
        )
    ], 
    outputs=[
        gr.Markdown(label="output"),
        gr.Image(show_label=False),
    ],
    title="Create your own GGUF Quants, blazingly fast ⚡!",
    description="Create GGUF quants from any Hugging Face repository! You need to specify a write token obtained in https://hf.co/settings/tokens.",
    article="<p>Find your write token at <a href='https://huggingface.co/settings/tokens' target='_blank'>token settings</a></p>",
    
)

# Launch the interface
iface.launch(debug=True)