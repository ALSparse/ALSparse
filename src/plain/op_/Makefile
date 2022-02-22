
SRC_DIR = $(shell find . -type d)

vpath %.c $(SRC_DIR)

SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.c) )

include $(ROOT)/Makefile.tail