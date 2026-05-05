# Master Makefile — D9 merge of cpp-engine-mxfp4 (D2 base + D3..D8 components)
#
# Builds ALL .cu in this directory into one libdsv4_engine_mxfp4.so
# linked against libcublas, libcudart, libdl. The 4 external kernel .so
# files (libfp8_e4m3_gemv, libmxfp4_gemv, libmxfp4_grouped_gemv,
# librmsnorm_fuse) are dlopen'd at runtime by D2's load_kernels(...).

NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_121a
CUDA_INC  ?= /usr/local/cuda/include
CUDA_LIB  ?= /usr/local/cuda/lib64

KERNEL_DIR = ../
FP8_DIR    = $(KERNEL_DIR)/fp8-dense
MXFP4D_DIR = $(KERNEL_DIR)/mxfp4-dense
MXFP4R_DIR = $(KERNEL_DIR)/mxfp4-routed
RMSN_DIR   = $(KERNEL_DIR)/rmsnorm-fuse

NVCC_FLAGS = -O3 --use_fast_math --extended-lambda -arch=$(ARCH) \
             -Xcompiler=-fPIC -Xcompiler=-Wall -lineinfo -std=c++17 \
             -I$(CUDA_INC) -I.

LDFLAGS = -L$(CUDA_LIB) -lcublas -lcudart -ldl \
          -Xlinker -rpath -Xlinker $(abspath $(FP8_DIR)) \
          -Xlinker -rpath -Xlinker $(abspath $(MXFP4D_DIR)) \
          -Xlinker -rpath -Xlinker $(abspath $(MXFP4R_DIR)) \
          -Xlinker -rpath -Xlinker $(abspath $(RMSN_DIR))

# All .cu files compile together. Order does not matter; nvcc resolves syms.
SRC = dsv4_engine_mxfp4.cu \
      mla_full.cu \
      hc.cu \
      routed.cu \
      hotset.cu \
      head.cu \
      graph.cu \
      dsv4_engine_mxfp4_wire.cu

OUT = libdsv4_engine_mxfp4.so

.PHONY: all clean check

all: $(OUT)

$(OUT): $(SRC) dsv4_engine_mxfp4_internal.h
	$(NVCC) $(NVCC_FLAGS) -shared $(SRC) -o $(OUT) $(LDFLAGS)
	@echo "[build] $(OUT) OK"
	@nm $(OUT) | grep -E " T (dsv4_mxfp4_|hotset_|hc_|mla_|dsv4_routed_|embed_forward|final_norm|rms_norm|lm_head|sample_greedy|capture_decode|replay_decode)" | head -30

# Verify all required ABI symbols are exported.
check: $(OUT)
	@echo "[check] D2 base symbols:"
	@nm $(OUT) | grep " T dsv4_mxfp4_" | awk '{print "  " $$3}'
	@echo "[check] D3 mla_full:"
	@nm $(OUT) | grep " T mla_" | awk '{print "  " $$3}'
	@echo "[check] D4 hc:"
	@nm $(OUT) | grep " T hc_" | awk '{print "  " $$3}'
	@echo "[check] D5 routed:"
	@nm $(OUT) | grep " T dsv4_routed_" | awk '{print "  " $$3}'
	@echo "[check] D6 hotset:"
	@nm $(OUT) | grep " T hotset_" | awk '{print "  " $$3}'
	@echo "[check] D7 head:"
	@nm $(OUT) | grep -E " T (embed_forward|final_norm|rms_norm|lm_head|sample_greedy)" | awk '{print "  " $$3}'
	@echo "[check] D8 graph:"
	@nm $(OUT) | grep -E " T (capture_decode|replay_decode|destroy_decode|alloc_static|alloc_io|free_engine_state)" | awk '{print "  " $$3}'

clean:
	rm -f $(OUT) *.o
