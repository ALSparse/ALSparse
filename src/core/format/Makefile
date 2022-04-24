
SRC_DIR = $(shell find . -type d)

vpath %.c $(SRC_DIR)

SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.c) )

ifeq ($(HIP_ON),1)
SRCCPP = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )
endif

include $(ROOT)/Makefile.tail
