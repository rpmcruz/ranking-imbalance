#!/bin/sh
cd cpp
g++ -g -Wall -c -fPIC neuralnet.cpp -o neuralnet.o
g++ -shared -Wl,-soname,libneuralnet.so -o libneuralnet.so neuralnet.o -larmadillo
