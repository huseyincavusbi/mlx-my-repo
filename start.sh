cd llama.cpp
make -j quantize gguf-split
cd ..
python app.py
