# Native entry point. For complete generation use: bash ../prepare_ssb.sh
.DEFAULT_GOAL := native
.PHONY: native loader legacy_loader
BUILD_DIR ?= ../../build/ssb-loader
native:
	cmake -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release
	cmake --build "$(BUILD_DIR)" --target ssb_convert
loader: native

# Historical encoded format only; never use it for a native dataset.
legacy_loader: load_modified.c
	gcc -o legacy_loader load_modified.c

original_loader: load.c
	gcc -o gpuDBLoader load.c

sort: columnSort.c
	gcc -o columnSort columnSort.c -std=c99 

rle: rle.c
	gcc -std=c99 rle.c -o rleCompression

dict: dict.c
	gcc -std=c99 dict.c -o dictCompression

clean:
	rm -rf *.o gpuDBLoader columnSort rleCompression dictCompression 
