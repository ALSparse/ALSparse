SRC_DIR = .
vpath %.cpp $(SRC_DIR)

TOOL_SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(TOOL_SRC) )

.PHONY: matrix_generator
matrix_generator : $(BIN_DIR)/matrix_generator


$(BIN_DIR)/matrix_generator:  $(OBJ)
	g++ -o $(BIN_DIR)/matrix_generator $^ -std=c++14

$(OBJ_DIR)/%.o : %.cpp
	g++ -c -o $@ $^ -std=c++14

%.cpp : %.hpp
	@touch $@