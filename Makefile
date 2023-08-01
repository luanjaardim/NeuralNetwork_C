
all: main
	./a

main:
	gcc main.c matrix.c nn.c -o a -lm
