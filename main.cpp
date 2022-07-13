#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include<windows.h>

const double WEIGHTS_UPPER_BOUND = -2;
const double WEIGHTS_LOWER_BOUND = 2;
const double BIAS_UPPER_BOUND = -2;
const double BIAS_LOWER_BOUND = 1;

using namespace std;

double randdouble(double lower_bound, double upper_bound) {
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::random_device rd;
	std::mt19937 gen(rd());
	return unif(gen);
}

static double actFunc(double x) {
	return 1/(1 + exp(-10*(x - 0.5)));
}

static double df(double x) {
	return actFunc(x)*(1 - actFunc(x))*10;
}

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

	void makeLayer() {
		neurons.resize(num_of_neurons);
		for (int i = 0; i < num_of_neurons; i++) {
			neurons[i].prev_value = 0;
			neurons[i].bias = randdouble(BIAS_LOWER_BOUND, BIAS_UPPER_BOUND);
			neurons[i].weights.resize(num_of_axons);
			for (int j = 0; j < num_of_axons; j++) {
				neurons[i].weights[j] = randdouble(WEIGHTS_LOWER_BOUND, WEIGHTS_UPPER_BOUND);
			}
		}
	}

	vector<double> getOutput(vector<double> x) {
		vector<double> y(num_of_neurons);
		for (int i = 0; i < num_of_neurons; i++) {
			double sum = 0;
			for (int j = 0; j < num_of_axons; j++) {
				sum += neurons[i].weights[j]*x[j];
			}
			sum += neurons[i].bias;
			neurons[i].prev_value = sum;
			y[i] = actFunc(sum);
		}
		return y;
	}

	vector<double> trainLayer(vector<double> de_dh) {
		for (int i = 0; i < num_of_neurons; i++) {
			double lr = -0.1;
			double de_dbi = de_dh[i]*df(neurons[i].prev_value);
			neurons[i].bias += 0.5*lr*de_dbi;
			double de_dw;
			for (int j = 0; j < num_of_axons; j++) {
				de_dw = de_dh[i]*df(neurons[i].prev_value)*neurons[i].weights[j];
				neurons[i].weights[j] += lr*de_dw;
			}
		}
		vector<double> de_dt(num_of_axons);
		for (int i = 0; i < num_of_axons; i++) {
			double cumsum = 0;
			for (int j = 0; j < num_of_neurons; j++) {
				double dhj_dti = neurons[j].weights[i]*df(neurons[j].prev_value);
				cumsum += de_dh[j]*dhj_dti;
			}
			de_dt[i] = cumsum;
		}
		return de_dt;
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

	double fit(vector<double> x, vector<double> y) {
		int size_of_ans = (int) y.size();
		vector<double> y_pred = predict(move(x));
		double mse = 0;
		for (int i = 0; i < size_of_ans; i++) {
			mse += (y_pred[i] - y[i])*(y_pred[i] - y[i]);
		}
		mse /= size_of_ans;
		vector<double> de_dt(size_of_ans);
		// Calculation of the derivative of the error over the last layer
		for (int i = 0; i < size_of_ans; i++) {
			de_dt[i] = 2*(y_pred[i] - y[i])/(double) size_of_ans;
		}
		for (int i = num_of_layers - 1; i >= 0; i--) {
			de_dt = layers[i].trainLayer(de_dt);
		}
		return mse;
	}

	void shuffle() {
		for (int i = 0; i < num_of_layers; i++) {
			for (int k = 0; k < layers_sizes[i]; k++) {
				int num_of_axons = layers[i].num_of_axons;
				layers[i].neurons[k].bias = randdouble(BIAS_LOWER_BOUND, BIAS_UPPER_BOUND);
				for (int j = 0; j < num_of_axons; j++) {
					layers[i].neurons[k].weights[j] = randdouble(WEIGHTS_LOWER_BOUND, WEIGHTS_UPPER_BOUND);
				}
			}
		}
	}
};

int main() {
	MyNetwork network;
	vector<int> amount = {2, 1};
	network.makeLayers(amount);

	int examples = 4;
	vector<double> inps[] = {{0, 0},
	                         {0, 1},
	                         {1, 0},
	                         {1, 1}};
	vector<double> answs[] = {{0},
	                          {1},
	                          {1},
	                          {0}};

	for (int i = 0; i < 250; i++) {
		double mse = 10;
		int j = 0;
		network.shuffle();
		while (mse > 0.01 && j < 100) {
			mse = 0;
			for (int k = 0; k < examples*10; k++) {
				mse += network.fit(inps[k%examples], answs[k%examples]);
			}
			mse /= 10*examples;
			j++;
		}
		cout << "\nMSE: " << mse;
		if (mse <= 0.01) {
			break;
		}
	}
	cout << "\nPredictions: " << network.predict(inps[0])[0] << " "
	     << network.predict(inps[1])[0] << " "
	     << network.predict(inps[2])[0] << " "
	     << network.predict(inps[3])[0] << " ";
}
