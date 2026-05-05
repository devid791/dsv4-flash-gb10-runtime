# Makefile — FP8 E4M3 GEMV M=1 dense kernel ()
# Target: GB10 sm_121a (Blackwell, CUDA 13.2)
# Build:  make           → libfp8_e4m3_gemv.so (default tile N=32)
#         make sweep     → builds tile N=16, 32, 64, 128 variants for tuning
#         make clean

NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_121a
CUDA_INC  ?= /usr/local/cuda/include

NVCC_FLAGS = -O3 --use_fast_math --extended-lambda \
             -arch=$(ARCH) \
             -Xcompiler=-fPIC \
             -Xcompiler=-Wall \
             -lineinfo \
             -I$(CUDA_INC)

SRC = fp8_e4m3_gemv.cu
OUT = libfp8_e4m3_gemv.so

.PHONY: all sweep clean

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -shared $(SRC) -o $(OUT)

# Sweep variants: tile width over output rows per CTA (BLOCK_N).
# Built as separate .so files for binding to A/B test from Python.
sweep: $(OUT)
	$(NVCC) $(NVCC_FLAGS) -DFP8_GEMV_BLOCK_N=16  -shared $(SRC) -o libfp8_e4m3_gemv_n16.so
	$(NVCC) $(NVCC_FLAGS) -DFP8_GEMV_BLOCK_N=32  -shared $(SRC) -o libfp8_e4m3_gemv_n32.so
	$(NVCC) $(NVCC_FLAGS) -DFP8_GEMV_BLOCK_N=64  -shared $(SRC) -o libfp8_e4m3_gemv_n64.so
	$(NVCC) $(NVCC_FLAGS) -DFP8_GEMV_BLOCK_N=128 -shared $(SRC) -o libfp8_e4m3_gemv_n128.so

clean:
	rm -f libfp8_e4m3_gemv*.so
