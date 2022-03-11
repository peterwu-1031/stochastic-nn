CXX = g++
CFLAGS = -Wall -std=c++11 -Wno-psabi
OPTFLAGS = -g
# LDFLAGS = -L/usr/local/lib -Llib/
# OPTFLAGS = -O3

SRC_DIR = src/
INC_DIR = inc/ 
BIN_DIR = bin/
LIB_DIR = lib/

LIBS = 
EXEC = main
BIN	= $(addprefix $(BIN_DIR), $(EXEC))
INC = -I$(INC_DIR)
SRC = $(wildcard $(SRC_DIR)*.cpp)
OBJ = $(SRC:%.cpp=%.o)

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CFLAGS) $(OPTFLAGS) $^ $(LIBS) -o $(BIN)
	@echo "FINISH"

%.o : %.cpp
	@echo ">> compiling: $<"
	$(CXX) $(CFLAGS) $(INC) $(LIBS) -c $< -o $@
clean:
	rm -f $(BIN) $(OBJ)


