import dlib
import numpy as np

x1 = []
y1 = []
f = open("training.txt", "r")
for x in f:
    var = x.split()
    x1.append([var[0], var[1]])
    y1.append([var[2]])

X = np.array(x1, dtype=float)  # input
y = np.array(y1, dtype=float)  # output

# scale units
x = X / np.amax(X, axis=0)  # intoarce maxim din lista x


class NeuralNetwork(object):

    def __init__(self):
        self.inputSize = 2
        self.hiddenSize = 10
        self.outputSize = 1

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (2x10) hidden, input layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (10x1)

    def sigmoid(self, s, deriv=False):
        # ia o valoare si o transforma in probabilitate intre 0 si 1
        if (deriv == True):
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def feedForward(self, X):
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z)  # Functia De Activare
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)

        return output

    def backward(self, X, y, output):
        # eroare = diferenta dintre val dorita si cea primita
        self.output_error = y - output  # eroare in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(self.W2.T)  # z2 error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)  # derivata sigmoida cu z2 error

        self.W1 += X.T.dot(self.z2_delta)  # prima ajustare (input-hidden)
        self.W2 += self.z2.T.dot(self.output_delta)  # 2 ajustare (hidden -output)

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)


if __name__ == '__main__':
    NN = NeuralNetwork()
    #antrenament:
    for i in range(1000):  # trains the NN 1000 times
        for i in range(len(X) - 1):
            if (i % 100 == 0):
                print("Loss: " + str(np.mean(np.square(y[i] - NN.feedForward(X[i])))))
            NN.train(np.array([X[i]]), y[i])

    #testare:
    # left_blink = 0.20
    # right_blink = 0.19
    # print("Input: " + str(np.array(([left_blink, right_blink]), dtype=float)))
    # print("Loss: " + str(np.mean(np.square(y - NN.feedForward([left_blink, right_blink])))))
    # print("\n")
    # print("Predicted Output: " + str(NN.feedForward([left_blink, right_blink])))
    # print(NN.W1)
    # print(NN.W2)
    #ponderile care se modifica in fct de fiecare poza care trece prin retea, si se pun ponderile cand reteaua e gata antrenata
    with open('results_of_training1.csv', 'w') as my_file:
        for i in NN.W1:
            np.savetxt(my_file, i)

    with open('results_of_training2.csv', 'w') as my_file:
        for i in NN.W2:
            np.savetxt(my_file, i)
   # print(left_blink, right_blink)

