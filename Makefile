# CUDA Makefile - Matmul
# Written by Loko Kung, 2018
# Edited by Tyler Port, 2018, Utilized by Cody Rivera


# Input Names
CUDA_HEADERS = multiply.cuh kernels.cuh
CUDA_FILES = multiply.cu
CPP_FILES = main.cpp
# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
              --expt-relaxed-constexpr
NVCC_INCLUDE = 
NVCC_LIBS = 
NVCC_GENCODES = -arch=sm_50 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_70,code=compute_70

# CUDA Object Files
CUDA_OBJ = cuda.o

CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets -g

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcublas -lcudart

# ------------------------------------------------------------------------------
# Make Rules (Lab 3 specific)
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ_FILES = $(notdir $(addsuffix .o, $(CPP_FILES)))

# Top level rules
all: multiply print gen

multiply: $(OBJ_FILES) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o multiply $(INCLUDE) $^ $(LIBS) 

print: print.cpp.o
	$(GPP) $(FLAGS) -o print $(INCLUDE) $^ $(LIBS)

gen: gen.cpp.o
	$(GPP) $(FLAGS) -o gen $(INCLUDE) $^ $(LIBS)


# Compile C++ Source Files
%.cpp.o: %.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 

# Compile CUDA Source Files
%.cu.o: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f multiply *.o *~ *.test.* print gen

.PHONY: clean
