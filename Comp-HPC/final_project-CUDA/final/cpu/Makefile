# Variables
COMPILER = g++
LIBS = utils.cpp
FILENAME = main.cpp
FLAGS = -fopenmp -O3 -g
OUTPUT = cpu.out
ARGUMENTS = 506250000 0.0 6.55 0.886227

# Default target
all: $(OUTPUT)

# Compile target
$(OUTPUT): $(FILENAME)
	$(COMPILER) $(FILENAME) $(LIBS) -o $(OUTPUT)

# Run target
run: $(OUTPUT)
	./$(OUTPUT) $(ARGUMENTS) 0 > cpu_single.txt
	./$(OUTPUT) $(ARGUMENTS) 1 > cpu_multi.txt

# Clean target
clean:
	rm -f $(OUTPUT)

.PHONY: all clean run