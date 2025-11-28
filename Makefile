CXX = clang++
SDK_PATH = $(shell xcrun --sdk macosx --show-sdk-path)
CXXFLAGS = -std=c++20 -Wall -Wextra -g -DDEBUG -isysroot $(SDK_PATH) -I$(SDK_PATH)/usr/include/c++/v1 -Iinclude
DEPFLAGS = -MMD -MP
SRCDIR = src
SOURCES = $(SRCDIR)/main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(OBJECTS:.o=.d)
TARGET = knot-rl

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(DEPS)

.PHONY: all clean

-include $(DEPS)
