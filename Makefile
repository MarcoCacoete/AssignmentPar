assessment1: assessment1.cpp
	g++ -std=c++0x assessment1.cpp -o assessment1 -lOpenCL -lX11 -lpthread
clean:
	rm assessment1
