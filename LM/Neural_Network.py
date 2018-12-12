import numpy as np


class NN:
    def __init__(self, nn):
        """
        :param nn: Structure of the NN
        Example: [2, 3, 4, 1] represents a NN with
                two inputs
                two hidden layers with 3 and 4, respectively
                one linear output layer
        """
        self.net = {'nn': nn, 'M': len(nn) - 1, 'layers': nn[1:]}
        self.net['w'], self.net['b'], self.net['N'] = self.create_w()

    def create_w(self):
        w = []
        b = []
        weights = 0
        for i in range(self.net['M']):
            w.append(np.random.rand(self.net['nn'][i+1], self.net['nn'][i]) - 0.5)
            b.append(np.random.rand(self.net['nn'][i+1]))
            weights += self.net['nn'][i+1] * (self.net['nn'][i] + 1)

        return w, b, weights

    @staticmethod
    def sigmoid(x):
        return 1.0/(1 + np.exp(-x))

    """
    @staticmethod
    def sigmoid_prime(x):
        return NN.sigmoid(x)*(1 - NN.sigmoid(x))
    """

    def feed_forward(self, x):
        ret = x
        for i in range(len(self.net['w'])):
            ret = np.dot(ret, self.net['w'][i].transpose())
            for j in range(ret.shape[0]):
                ret[j][:] += self.net['b'][i]
            ret = NN.sigmoid(ret)
        return ret

    """
    def feed_forward1(self, x):
        ret = x
        activation = []
        y = []
        s = []
        for i in range(len(self.net['w'])):
            ret = np.dot(ret, self.net['w'][i].transpose())
            for j in range(ret.shape[0]):
                ret[j][:] += self.net['b'][i]
            activation.append(ret)
            s.append(NN.sigmoid_prime(ret))
            ret = NN.sigmoid(ret)
            y.append(ret)
        return ret, activation, y, s
    """

    @staticmethod
    def map_weights(w, b):
        layer_to_w = {}
        w_to_layer = {}
        cnt = 0
        for layer in range(len(w)):
            for i in range(w[layer].shape[0]):
                layer_to_w[(layer, i, -1)] = cnt
                w_to_layer[cnt] = (layer, i, -1)
                cnt += 1
                for j in range(w[layer].shape[1]):
                    layer_to_w[(layer, i, j)] = cnt
                    w_to_layer[cnt] = (layer, i, j)
                    cnt += 1
        return layer_to_w, w_to_layer

    def train_lm(self, x, y, min_error=0.01, epochs=100, u=0.01):
        patterns = len(x)
        output_neurons = self.net['nn'][len(self.net['nn']) - 1]
        layer_to_w, w_to_layer = NN.map_weights(self.net['w'], self.net['b'])
        prev_error = 100
        m = 0
        for epoch in range(epochs):
            prediction = self.feed_forward(x)
            e = NN.create_e(patterns, output_neurons, y, prediction)
            error = 0.5 * np.sum(np.square(e))
            if error <= min_error:
                break
            j = self.create_j(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)
            tmp = np.dot(np.linalg.pinv(np.dot(j.transpose(), j) + u*np.identity(j.shape[1])), np.dot(j.transpose(), e.transpose()))

            tmp_w = self.net['w']
            tmp_b = self.net['b']

            for i in range(len(tmp)):
                tup = w_to_layer[i]
                if tup[2] == -1:
                    self.net['b'][tup[0]][tup[1]] -= tmp[i]
                else:
                    self.net['w'][tup[0]][tup[1]][tup[2]] -= tmp[i]

            e = NN.create_e(patterns, output_neurons, y, prediction)
            error = 0.5 * np.sum(np.square(e))
            if error >= prev_error:
                u *= 10
                if m <= 5:
                    self.net['w'] = tmp_w
                    self.net['b'] = tmp_b
                    m += 1
                    continue
            else:
                m = 0
                #u /= 10
            prev_error = error

        return self.net['w'], self.net['b']

    """
    def __train_lm1(self, x, y, min_error=0.01, epochs=100, u=0.01):
        patterns = len(x)
        output_neurons = self.net['nn'][len(self.net['nn']) - 1]
        layer_to_w, w_to_layer = NN.map_weights(self.net['w'], self.net['b'])
        epoch = 0
        while True:
            epoch += 1
            prediction = self.feed_forward(x)
            e = NN.create_e(patterns, output_neurons, y, prediction)
            j = self.create_j1(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)
            prev_error = 0.5*np.sum(np.square(e))
            error = prev_error + 1
            m = 0
            while error > prev_error and m <= 5:
                tmp = np.dot(np.linalg.pinv(np.dot(j.transpose(), j) + u * np.identity(j.shape[1])), np.dot(j.transpose(), e.transpose()))

                tmp_w = self.net['w']
                tmp_b = self.net['b']

                for i in range(len(tmp)):
                    tup = w_to_layer[i]
                    if tup[2] == -1:
                        self.net['b'][tup[0]][tup[1]] -= tmp[i]
                    else:
                        self.net['w'][tup[0]][tup[1]][tup[2]] -= tmp[i]

                prediction = self.feed_forward(x)
                e = NN.create_e(patterns, output_neurons, y, prediction)
                error = 0.5 * np.sum(np.square(e))
                print(error)
                if error <= min_error:
                    break
                if error <= prev_error:
                    u /= 10
                    break
                m += 1
                u *= 10
                self.net['w'] = tmp_w
                self.net['b'] = tmp_b
            if error <= min_error:
                break
                
        ""for epoch in range(epochs):
            prediction = self.feed_forward(x)
            e = NN.create_e(patterns, output_neurons, y, prediction)
            j = self.create_j1(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)
            tmp = np.dot(np.linalg.pinv(np.dot(j.transpose(), j) + u*np.identity(j.shape[1])), np.dot(j.transpose(), e.transpose()))

            tmp_w = self.net['w']
            tmp_b = self.net['b']

            for i in range(len(tmp)):
                tup = w_to_layer[i]
                if tup[2] == -1:
                    self.net['b'][tup[0]][tup[1]] -= tmp[i]
                else:
                    self.net['w'][tup[0]][tup[1]][tup[2]] -= tmp[i]

            e = NN.create_e(patterns, output_neurons, y, prediction)
            error = 0.5 * np.sum(np.square(e))
            if error <= min_error:
                break
            if error > prev_error:
                u *= 10
                if m <= 5:
                    self.net['w'] = tmp_w
                    self.net['b'] = tmp_b
                    m += 1
                    continue
                else:
                    m = 0
                    #u /= 10
            else:
                m = 0
                #u /= 10
            prev_error = error""

        return self.net['w'], self.net['b']
    
    def test(self, x, y):
        patterns = len(x)
        output_neurons = self.net['nn'][len(self.net['nn']) - 1]
        prediction = self.feed_forward(x)
        layer_to_w, w_to_layer = NN.map_weights(self.net['w'], self.net['b'])
        e = NN.create_e(patterns, output_neurons, y, prediction)
        j = self.create_j(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)
        j1 = self.create_j1(x, y, e, self.net['N'], layer_to_w, w_to_layer, patterns, output_neurons)

        print(np.sum(np.subtract(j, j1)))
    """

    @staticmethod
    def create_e(patterns, output_neurons, y, prediction):
        e = np.zeros([1, patterns*output_neurons])
        cnt = 0
        for i in range(patterns):
            for j in range(output_neurons):
                e[0][cnt] = y[i][j] - prediction[i][j]
                cnt += 1

        return e

    def create_j(self, x, y, e, weights, layer_to_w, w_to_layer, patterns, output_neurons):
        delta_w = 0.001
        j = np.zeros([e.shape[1], weights])
        original_e = NN.create_e(patterns, output_neurons, y, self.feed_forward(x)).transpose()

        for i in range(weights):
            tup = w_to_layer[i]
            if tup[2] == -1:
                self.net['b'][tup[0]][tup[1]] += delta_w
            else:
                self.net['w'][tup[0]][tup[1]][tup[2]] += delta_w

            predict_e = NN.create_e(patterns, output_neurons, y, self.feed_forward(x)).transpose()
            subt = np.subtract(predict_e, original_e)
            for row in range(j.shape[0]):
                j[row][i] += subt[row]

            if tup[2] == -1:
                self.net['b'][tup[0]][tup[1]] -= delta_w
            else:
                self.net['w'][tup[0]][tup[1]][tup[2]] -= delta_w
        return j

    """
    def create_j1(self, x, y, e, weights, layer_to_w, w_to_layer, patterns, output_neurons):
        j = np.zeros([e.shape[1], weights])
        prediction, activation, t, s = self.feed_forward1(x)

        delta = []
        for layer in range(self.net['M']):
            delta.append(np.zeros([output_neurons, self.net['nn'][layer + 1]]))

        all_delta = []  # pattern / layer / to_output / neuron_in_layer
        for i in range(len(x)):
            all_delta.append(delta)

        for layer in range(self.net['M'] - 1, -1, -1):
            if layer == self.net['M'] - 1:
                for ii in range(delta[layer].shape[0]):
                    for jj in range(delta[layer].shape[1]):
                        if ii == jj:
                            delta[layer][ii][jj] = s[layer][jj]
            else:
                for o in range(output_neurons):
                    for k in range(delta[layer].shape[1]):
                        for i in range(self.net['nn'][layer+2]):
                            delta[layer][o][k] += self.net['w'][layer+1][o][i] * delta[layer+1][o][i]
                for o in range(output_neurons):
                    for k in range(delta[layer].shape[1]):
                        for m in range(len(x)):
                            all_delta[m][layer][o][k] = delta[layer][o][k] * s[layer][m][k]

        for p in range(len(x)):
            for m in range(output_neurons):
                cnt = 0
                for layer in range(self.net['M']):
                    for jj in range(self.net['w'][layer].shape[0]):
                        for ii in range(self.net['w'][layer].shape[1]):
                            if layer > 0:
                                j[p * 0 + m][cnt] = -(all_delta[p][layer][m][jj] * t[layer-1][m][ii])
                            else:
                                j[p * 0 + m][cnt] = -(all_delta[p][layer][m][jj] * float(x[m][ii]))
                            cnt += 1

        return j
    """

    def predict(self, x):
        predict = self.feed_forward(x)
        #return predict
        for i in range(predict.shape[0]):
            for j in range(predict.shape[1]):
                if predict[i][j] >= 0.5:
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0
        return predict
