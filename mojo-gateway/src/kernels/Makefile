# T-MAC Kernel Build System
# Builds shared libraries for x86 (AVX2) and ARM (NEON)

CC = clang
CFLAGS_COMMON = -O3 -fPIC -Wall -Wextra

# Platform detection
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),x86_64)
    CFLAGS_ARCH = -mavx2 -mfma
    PLATFORM = x86_64
else ifeq ($(UNAME_M),arm64)
    CFLAGS_ARCH =
    PLATFORM = arm64
else ifeq ($(UNAME_M),aarch64)
    CFLAGS_ARCH =
    PLATFORM = arm64
else
    CFLAGS_ARCH =
    PLATFORM = generic
endif

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT = dylib
    SHARED_FLAGS = -dynamiclib
else
    SHARED_EXT = so
    SHARED_FLAGS = -shared
endif

CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_ARCH)

# Output directories
LIB_DIR = ../../lib
BIN_DIR = ../../bin

# Targets
TARGET = $(LIB_DIR)/libtmac_kernel.$(SHARED_EXT)
STATIC_TARGET = $(LIB_DIR)/libtmac_kernel.a

SOURCES = tmac_kernel.c
OBJECTS = tmac_kernel.o

.PHONY: all clean test install dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(LIB_DIR) $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(CC) $(SHARED_FLAGS) -o $@ $^ -lm
	@echo "Built $(TARGET) for $(PLATFORM)"

$(STATIC_TARGET): $(OBJECTS)
	ar rcs $@ $^
	@echo "Built static library $(STATIC_TARGET)"

%.o: %.c tmac_kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

# Test binary
test: dirs tmac_kernel.o test_kernel.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/test_kernel test_kernel.c tmac_kernel.o -lm
	@echo "Running tests..."
	$(BIN_DIR)/test_kernel

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET) $(STATIC_TARGET)
	rm -f $(BIN_DIR)/test_kernel

# Install to system (optional)
install: $(TARGET)
	@echo "Installing to /usr/local/lib..."
	cp $(TARGET) /usr/local/lib/
	cp tmac_kernel.h /usr/local/include/

# Cross-compilation targets
arm64:
	$(MAKE) CC=aarch64-linux-gnu-gcc CFLAGS_ARCH="" PLATFORM=arm64

x86:
	$(MAKE) CC=clang CFLAGS_ARCH="-mavx2 -mfma" PLATFORM=x86_64

# Raspberry Pi specific
rpi:
	$(MAKE) CC=arm-linux-gnueabihf-gcc CFLAGS_ARCH="-mfpu=neon" PLATFORM=arm32

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: all

# Benchmark build
benchmark: CFLAGS += -DBENCHMARK
benchmark: all

# Print configuration
info:
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "Target: $(TARGET)"
