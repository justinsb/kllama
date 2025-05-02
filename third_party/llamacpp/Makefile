build-llama-cpp:
	cd llama.cpp && cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF
	cd llama.cpp && cmake --build build --config Release -j 8
