# NeuralNetwork_C
My first try to implementing a neural network, obviously in C.

# Using the NN
To use it, you can run:

make < train_xnor_and_xor.txt

To train your own model you need to undef DEBUG and pass the desired layout to the *nn_create* function and then save the parameters:

gcc main.c matrix.c nn.c -o a -lm
./a > new_train.txt
