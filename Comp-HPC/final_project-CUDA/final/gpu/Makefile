# Variables
COMPILER = nvcc
FLAGS = -rdc=true
LIBS = -lcurand
FILENAME = mc-reduction.cu
COMPLEMENTS = lib.cpp
OUTPUT = cuda.x
COMPILER_PATH = -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe"
ARGUMENTS = 22500 1234 32 0.0 6.55 0.886227 f_x.txt

# Default target
all: $(OUTPUT)

# Compile target
$(OUTPUT): $(FILENAME)
	$(COMPILER) $(FLAGS) $(FILENAME) $(COMPLEMENTS) $(LIBS) $(COMPILER_PATH) -o $(OUTPUT)

# Run target
run: $(OUTPUT)
	./$(OUTPUT) $(ARGUMENTS)

# Clean target
clean:
	rm -f $(OUTPUT)

.PHONY: all clean run