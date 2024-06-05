# Variables
COMPILER = nvcc
LIBS = -lcurand
FILENAME = curand-test.cu
FLAGS = -extended-lambda
OUTPUT = rand.exe
COMPILER_PATH = -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe"
ARGUMENTS = 100 1234 32 0.0 6.55
OMP = g++ -fopenmp -g -fsanitize=thread montecarlo-parallel_for.cpp
# Default target
all: $(OUTPUT)

# Compile target
$(OUTPUT): $(FILENAME)
	$(COMPILER) $(FILENAME) $(LIBS) $(COMPILER_PATH) -o $(OUTPUT)

# Run target
run: $(OUTPUT)
	./$(OUTPUT) $(ARGUMENTS)

# Clean target
clean:
	rm -f $(OUTPUT)

.PHONY: all clean run