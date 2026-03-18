class ZNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        # Perform a weighted sum of the inputs, add the bias
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return total

    def __repr__(self):
        return f"ZNeuron({self.weights}, {self.bias})"
    

def test():
    # Create a neuron
    neuron = ZNeuron([1, 2, 3], 4)
    # Create some inputs
    inputs = [5, 6, 7]
    # Perform forward pass
    output = neuron.forward(inputs)
    # Print the output
    print(output)
    # Print the neuron
    print(neuron)
    