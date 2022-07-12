#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;

class Neuron {
public:
	vector<double> weights;
	double bias;
	double prev_value;
};

class Layer {
public:
	int num_of_neurons;
	vector<Neuron> neurons;
	int num_of_axons;

	static double actFunc(double x) {
		return 1/(1 + exp(-x));
	}

	static double df(double x) {
		return actFunc(x)*(1 - actFunc(x));
	}

	void makeLayer() {
		neurons.resize(num_of_neurons);
		for (int i = 0; i < num_of_neurons; i++) {
			neurons[i].prev_value = 0;
			neurons[i].bias = 0;
			neurons[i].weights.resize(num_of_axons);
			for (int j = 0; j < num_of_axons; j++) {
				neurons[i].weights[j] = 2.0*rand()/(double) RAND_MAX - 1;
			}
		}
	}

	vector<double> getOutput(vector<double> x) {
		vector<double> y(num_of_neurons);
		for (int i = 0; i < num_of_neurons; i++) {
			double sum = 0;
			for (int j = 0; j < num_of_axons; j++) {
				sum += neurons[i].weights[j]*x[i];
			}
			sum += neurons[i].bias;
			neurons[i].prev_value = sum;
			y[i] = actFunc(sum);
		}
		return y;
	}
};

class MyNetwork {
public:
	int num_of_layers{};
	vector<int> layers_sizes;
	vector<Layer> layers;

	void makeLayers(vector<int> n_neurons) {
		num_of_layers = (int) n_neurons.size();
		layers_sizes = move(n_neurons);
		layers.resize(num_of_layers);
		layers[0].num_of_neurons = layers_sizes[0];
		layers[0].num_of_axons = layers_sizes[0];
		layers[0].makeLayer();
		for (int i = 1; i < num_of_layers; i++) {
			layers[i].num_of_neurons = layers_sizes[i];
			layers[i].num_of_axons = layers_sizes[i - 1];
			layers[i].makeLayer();
		}
	}

	vector<double> predict(vector<double> x) {
		vector<double> y = move(x);
		for (int i = 0; i < num_of_layers; i++) {
			y = layers[i].getOutput(y);
		}
		return y;
	}
};

int main() {
	MyNetwork network;
	vector<int> amount = {2, 3, 1};
	network.makeLayers(amount);

	vector<double> x = {0, 1};
	vector<double> answers = network.predict(x);
	int ans_size = (int) answers.size();
	for (int i = 0; i < ans_size; i++) {
		cout << answers[i] << " ";
	}
}
