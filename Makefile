CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
CFLAGS += -O3 -s -DNDEBUG -std=c++11

all:
	nvcc -I. -arch=sm_52 -c src/bilateral_gpu.cu -o bilateral_gpu.o
	g++ -o bilateral_filter src/main.cpp src/bilateral_cpu.cpp bilateral_gpu.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart -I/usr/include/ -Xlinker -Map=bilateral_filter.map 

clean: 
	@rm -rf *.o build/
	@rm -rf *~
