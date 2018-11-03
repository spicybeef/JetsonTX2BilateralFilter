CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
CFLAGS += -O3 -s -DNDEBUG

all:
	#nvcc -I. -arch=sm_52 -c src/bilateral_gpu.cu -o build/bilateral_gpu.o 
	g++ -o bilateral_filter src/main.cpp $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart -I/usr/include/

clean: 
	@rm -rf *.o build/
	@rm -rf *~
