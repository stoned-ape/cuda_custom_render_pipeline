a.exe: main.cu makefile
	nvcc -lX11 -Iglm main.cu --library cuda -o a.exe

run: a.exe
	./a.exe