CC     ?= gcc
CFLAGS := -O3 -march=native -mtune=native -fopenmp
LDLIBS := -lm

all: roofline

roofline: roofline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f roofline

.PHONY: all clean
