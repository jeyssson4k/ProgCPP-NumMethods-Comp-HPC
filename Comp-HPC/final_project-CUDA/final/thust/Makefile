# Variables
COMPILER = nvcc
FILENAME = thrust-double.cu
OUTPUT = thrust.exe
COMPILER_PATH = -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe"
ARGUMENTS = 22500 0.886227

# Default target
all: $(OUTPUT)

# Compile target
$(OUTPUT): $(FILENAME)
	$(COMPILER) $(FILENAME) $(COMPILER_PATH) -o $(OUTPUT)

# Run target
run: $(OUTPUT)
	./$(OUTPUT) $(ARGUMENTS)

# Clean target
clean:
	rm -f $(OUTPUT)

.PHONY: all clean run