cd llama.cpp
LLAMA_CUDA=1 make -j llama-quantize llama-gguf-split llama-imatrix

cd ..
python app.py