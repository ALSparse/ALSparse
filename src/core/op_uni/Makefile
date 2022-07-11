
SRC_DIR = $(shell find . -type d)

vpath %.cpp $(SRC_DIR)

SRCCPP = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )

include $(ROOT)/Makefile.tail
