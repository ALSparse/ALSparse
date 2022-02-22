
#CPPFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=1

SRC_DIR = $(shell find . -type d)

vpath %.cpp $(SRC_DIR)
vpath %.S $(SRC_DIR)

SRCCPP = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )
ASM_SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.S) )

include $(ROOT)/Makefile.tail
