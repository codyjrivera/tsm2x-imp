CXX       := g++
NVCC      := nvcc
STD       := -std=c++11
CCFLAGS   := $(STD) -O3 -g
NVCCFLAGS := $(STD) -O3 -g
SRC_DIR   := src
OBJ_DIR   := obj

LINKFLAGS := -Wno-deprecated-gpu-targets

# Different Architecture Optimizations
# You may need to modify this for your own GPU setup
DEPLOY    := -arch=sm_50 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_70,code=compute_70
#	-gencode=arch=compute_50,code=sm_50 
#	-gencode=arch=compute_52,code=sm_52 
#	-gencode=arch=compute_60,code=sm_60 
#	-gencode=arch=compute_61,code=sm_61 

MAIN      := $(SRC_DIR)/multiply.cu
CUFILES2  := $(SRC_DIR)/kernels.cu
CUFILES1  := $(filter-out $(MAIN), $(CUFILES2) $(wildcard $(SRC_DIR)/*.cu))

CUOBJS2   := $(CUFILES2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS1   := $(CUFILES1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

OBJS      := $(CUOBJS1) $(CUOBJS2)

$(CUOBJS2): NVCCFLAGS += -rdc=true $(DEPLOY)
$(CUOBJS1): NVCCFLAGS +=

all: ; @$(MAKE) multiply gen print -j

multiply: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -lcublas -lcudart $(DEPLOY) $(LINKFLAGS) $(MAIN) -rdc=true $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

gen: gen.cpp
	$(CXX) $(CCFLAGS) $< -o $@

print: print.cpp
	$(CXX) $(CCFLAGS) $< -o $@

.PHONY: clean

clean:
	$(RM) $(OBJS) multiply gen print
