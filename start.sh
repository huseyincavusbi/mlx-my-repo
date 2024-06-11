cd llama.cpp
LLAMA_CUDA=1 make -j quantize gguf-split imatrix

cd ..
python app.py