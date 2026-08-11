CC=gcc
CFLAGS=-O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wno-unused-function -Wno-unused-variable
LDFLAGS=-lm -fopenmp
disco: nn.c archs.c main.c mem.c nn.h model.h mem.h
	$(CC) $(CFLAGS) -o disco nn.c archs.c main.c mem.c $(LDFLAGS)
clean:
	rm -f disco
