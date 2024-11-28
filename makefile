# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-O3 -arch=sm_90a --use_fast_math -std=c++17 -Ideps/cutlass/include --expt-relaxed-constexpr

# Target and source
TARGET=test
SRC=test.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean target
clean:
	rm -f $(TARGET)