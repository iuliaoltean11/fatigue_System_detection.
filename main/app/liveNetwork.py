import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NeuralNetwork(object):

    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 10

        self.W1 = np.loadtxt('results_of_training1.csv') #iei din fisier ponderile pentru retea preantrenata
        self.W1 = self.W1.reshape(self.inputSize, self.hiddenSize)

        self.W2 = np.loadtxt('results_of_training2.csv')
        self.W2 = self.W2.reshape(self.hiddenSize, self.outputSize)

    def sigmoid(self, s, deriv=False):
        """
        functie de activare care transforma fiecare rezultat al neuronului
        intre 0 si 1.

        """
        if (deriv == True):
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def feedForward(self, X):
        """
         se primesc valorile din stratul de intrare si se inmultesc cu ponderile w1 si w2
         se calcularea output.
        """
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.sigmoid(self.z3)
        return output


class SVM_Model():

    def __init__(self):
        dataset = pd.read_csv('SVM_training_data.csv')

        # split dataset in x si target_values
        training_vectors = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values
        target_values = dataset.iloc[:, 10].values

        # perform feature scaling
        self.scaler = StandardScaler()
        training_vectors = self.scaler.fit_transform(training_vectors)

        # fit svm to data
        self.classifier = SVC(kernel='rbf', random_state=0)
        self.classifier.fit(training_vectors, target_values)

    def predict_class(self, new_data):

        new_data = np.array(new_data).reshape(1, -1)
        new_data = self.scaler.transform(new_data)
        predicted_class = self.classifier.predict(new_data)[0]
        return predicted_class

if __name__ == '__main__':
    NN = NeuralNetwork()
    SVM = SVM_Model()
