CXX = clang++
SDK_PATH = $(shell xcrun --sdk macosx --show-sdk-path)
CXXFLAGS = -std=c++20 -Wall -Wextra -g -DDEBUG -isysroot $(SDK_PATH) -I$(SDK_PATH)/usr/include/c++/v1 -Iinclude
LDFLAGS =
LIBTORCH ?= $(abspath libtorch)
ifeq ($(wildcard $(LIBTORCH)),)
$(warning LIBTORCH directory not found at $(LIBTORCH); download/extract libtorch first)
endif
CXXFLAGS += -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include
LDFLAGS  += -L$(LIBTORCH)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH)/lib
DEPFLAGS = -MMD -MP
SRCDIR = src
SOURCES = $(SRCDIR)/main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(OBJECTS:.o=.d)
TARGET = knot-rl

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(DEPS)

.PHONY: all clean

-include $(DEPS)
