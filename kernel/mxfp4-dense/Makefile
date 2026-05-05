# Makefile — MXFP4 GEMV M=1 dense kernel # Target: GB10 sm_121a (Blackwell, CUDA 13.2)
# Build:  make           → libmxfp4_gemv.so (default tile N=64)
#         make sweep     → builds tile N=1, 4, 8, 32, 64, 128 variants for tuning
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

SRC = mxfp4_gemv.cu
OUT = libmxfp4_gemv.so

.PHONY: all sweep clean

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -shared $(SRC) -o $(OUT)

# Sweep variants: tile width over output rows per CTA (BLOCK_N).
# Built as separate .so files for binding to A/B test from Python.
sweep: $(OUT)
	for n in 1 4 8 32 128; do \
	  $(NVCC) $(NVCC_FLAGS) -DMXFP4_GEMV_BLOCK_N=$$n -shared $(SRC) -o libmxfp4_gemv_n$$n.so ; \
	done

clean:
	rm -f $(OUT) libmxfp4_gemv_n*.so
