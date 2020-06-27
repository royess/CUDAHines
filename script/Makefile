CC=gcc
CXX=g++

all: serial parallel

serial: ../src/serial.cc
	g++ -O2 -std=c++11 ../src/serial.cc -o ../bin/serial

parallel: ../src/parallel.cu
	nvcc -O2 ../src/parallel.cu -o ../bin/parallel

clean:
	rm -f ../bin/serial ../bin/parallel
