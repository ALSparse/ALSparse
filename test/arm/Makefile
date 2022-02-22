SRC_DIR = .
vpath %.c $(SRC_DIR)

TEST_SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_SRC) )
TEST_BIN = $(patsubst $(SRC_DIR)/%.c, $(BIN_DIR)/%, $(TEST_SRC) )

CFLAGS += $(INC) $(LIB) $(DEFINE) -l:libalphasparse.a -lm

include $(ROOT)/Makefile.tail
