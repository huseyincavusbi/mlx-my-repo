cd llama.cpp
make -j quantize gguf-split imatrix

cd ..
python app.py