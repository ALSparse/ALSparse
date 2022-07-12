
SRC_DIR = .
vpath %.c $(SRC_DIR)

TOOL_SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst %.c, $(OBJ_DIR)/%.o, $(TOOL_SRC) )
TOOL_BIN = $(patsubst %.c, $(BIN_DIR)/%, $(TOOL_SRC) )

CFLAGS = -w -g -O2 -std=c11 -lm
.SECONDEXPANSION:
# include $(ROOT)/Makefile.tail
.PHONY: matrix_generator
tool :: $(TOOL_BIN) matrix_generator

$(OBJ) : $(OBJ_DIR)/%.o : %.c
	$(CC) -c -o $@ $^ $(CFLAGS) -UONAME 

$(BIN_DIR)/% : $(OBJ_DIR)/%.o
	$(CC) -o $@ $^ $(CFLAGS)  

matrix_generator : 
	@make -C matrix_generator 