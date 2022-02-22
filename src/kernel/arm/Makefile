
CFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=1

SRC_DIR = $(shell find . -type d)

vpath %.c $(SRC_DIR)
vpath %.S $(SRC_DIR)

SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.c) )
ASM_SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.S) )

include $(ROOT)/Makefile.tail
