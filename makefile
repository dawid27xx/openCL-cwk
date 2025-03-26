# Makefile

CC = gcc
CFLAGS = -Wall -framework OpenCL
TARGET = cwk3
SRC = cwk3.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
