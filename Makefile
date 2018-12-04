CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
CFLAGS += -O3 -s -std=c++11 -fopenmp

all:
	nvcc -I. -arch=sm_52 -c src/bilateral_gpu.cu -o bilateral_gpu.o
	nvcc --source-in-ptx -ptx -G -I. -arch=sm_52 -c src/bilateral_gpu.cu
	g++ -o bilateral_filter src/main.cpp src/bilateral_cpu.cpp bilateral_gpu.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart -I/usr/include/
	g++ -c -g -S -fverbose-asm -O src/main.cpp src/bilateral_cpu.cpp $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart -I/usr/include/ -Wa,-aslh

clean: 
	@rm -rf *.o build/
	@rm -rf *~
