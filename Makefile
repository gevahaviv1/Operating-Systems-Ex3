CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDES := -Iinclude
SRCS     := $(wildcard src/*.cpp)
OBJS     := $(SRCS:src/%.cpp=build/%.o)
LIB      := libMapReduceFramework.a

all: $(LIB)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIB): $(OBJS)
	ar rcs $@ $^

build:
	mkdir -p build

example/SampleClient: $(LIB) example/SampleClient.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) example/SampleClient.cpp $(LIB) -o $@ -pthread

.PHONY: clean example
clean:
	rm -rf build $(LIB) example/SampleClient