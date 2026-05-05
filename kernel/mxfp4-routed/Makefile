# Makefile — kernel/mxfp4-routed # Builds libmxfp4_grouped_gemv.so for sm_121a Blackwell (GB10) GX10.

NVCC ?= nvcc
ARCH ?= sm_121a

CFLAGS  = -O3 --use_fast_math -std=c++17 -Xcompiler=-fPIC
ARCHFLAGS = -arch=$(ARCH)

SRC = mxfp4_grouped_gemv.cu
OBJ = mxfp4_grouped_gemv.o
LIB = libmxfp4_grouped_gemv.so

.PHONY: all clean

all: $(LIB)

$(OBJ): $(SRC)
	$(NVCC) $(ARCHFLAGS) $(CFLAGS) -dc $< -o $@

$(LIB): $(OBJ)
	$(NVCC) $(ARCHFLAGS) -shared $< -o $@

clean:
	rm -f $(OBJ) $(LIB)
