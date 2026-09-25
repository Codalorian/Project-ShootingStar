CFLAGS = -O2 -Wall -Wextra -std=c11 -pthread
all: ssd_roofline
ssd_roofline: ssd_roofline.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f ssd_roofline
.PHONY: all clean
