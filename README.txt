Big Thanks to Wilhelm at 3cglabs
Code mostly copied from the article on
trapexit.org/Erlang_and_Neural_Networks

Modified by Tyler Green 

Overview:  Experimental Asychronous Artificial Neural Network written
in erlan

To Do:

Implement trainer:
Trainer takes a list of examples, feed them to the network , one at a
time, and for each valid output, find the differecne with the training
value and sent it to the learn message of the output node.

Right now, 1 is just the default training value in {stimulate ... }
message

The current implementation also does not take explicit account of biases for every node. A bias is simply a weight in a node that is not associated with any particular input, but is added to the dot product before being sent through the sigmoid function. It does affect the expressiveness of the neural network, so I will also leave that as an exercise for the reader. 

