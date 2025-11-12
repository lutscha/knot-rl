CXX = clang++
SDK_PATH = $(shell xcrun --sdk macosx --show-sdk-path)
CXXFLAGS = -std=c++20 -Wall -Wextra -isysroot $(SDK_PATH) -I$(SDK_PATH)/usr/include/c++/v1
SRCDIR = src
SOURCES = $(SRCDIR)/knot.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = knot

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean

