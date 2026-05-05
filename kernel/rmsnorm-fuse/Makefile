# Makefile for librmsnorm_fuse.so
# Target: GB10 sm_121a (CUDA 13.2 container)

NVCC      ?= nvcc
ARCH      ?= sm_121a
TARGET    := librmsnorm_fuse.so
SRC       := rmsnorm_fuse.cu

NVCCFLAGS := -O3 -arch=$(ARCH) --use_fast_math \
             -Xcompiler -fPIC --threads 1 \
             -ftemplate-depth=256 --expt-relaxed-constexpr \
             -shared

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "[build] $@ OK"
	@nm -D $@ | grep -E "rmsnorm_fuse_(fwd|residual_add)" | head -4

clean:
	rm -f $(TARGET)
