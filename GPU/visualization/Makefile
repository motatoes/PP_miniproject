################################################################################
#
# Build script for project
#
################################################################################

# Executable name
EXECUTABLE	:= smb

# CUDA source files (compiled with nvcc)
CUFILES		:= main.cu

# CUDA dependency files
CU_DEPS		:= Makefile

# C/C++ source files (compiled with gcc/g++)
CCFILES		:= 
		   
# Libraries
USEGLLIB        := 1
USEGLUT         := 1

################################################################################
# Rules and targets


ROOTDIR = /bham/pd/packages/SL6/x86_64/cuda-4.0.17-sdk/C/common
BINDIR  = ./bin
include ../common.mk


# flag to enable double precision
# NVCCFLAGS += -arch sm_13 -lcurand

# flag to save intermediate files to look at PTX code
# NVCCFLAGS += -keep

# flag to clean up afterwards
# NVCCFLAGS += -keep -clean

# flag to report on use of registers, shared memory, etc
# NVCCFLAGS += -Xptxas -v

