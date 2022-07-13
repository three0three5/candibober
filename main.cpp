#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include<windows.h>

using namespace std;

double randdouble(double lower_bound, double upper_bound) {
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re;
	return unif(re);
}

static double actFunc(double x) {
	return x>0? x:0;
}

static double df(double x) {
	return x>0? 1:0;
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
			neurons[i].bias = 0;
			neurons[i].weights.resize(num_of_axons);
			for (int j = 0; j < num_of_axons; j++) {
				neurons[i].weights[j] = randdouble(0.0, 3.0);
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
			neurons[i].bias += 0.01*lr*de_dbi;
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
};

int main() {
	MyNetwork network;
	vector<int> amount = {1};
	network.makeLayers(amount);
	vector<double> x = {0};
	vector<double> answers = network.predict(x);
	int ans_size = (int) answers.size();
	for (int i = 0; i < ans_size; i++) {
		cout << answers[i] << " ";
	}
	int examples = 4;
	vector<double> inps[] = {{0},
	                         {1},
	                         {2},
	                         {4}};
	vector<double> answs[] = {{0},
	                          {2},
	                          {4},
	                          {8}};
	double mse = 1;
	while (mse > 0.01) {
		mse = 0;
		for (int i = 0; i < examples*10; i++) {
			mse += network.fit(inps[i%examples], answs[i%examples]);
		}
		mse /= 10*examples;
		cout << "\nMSE: " << mse;
	}
	cout << "\nPredictions: " << network.predict(inps[0])[0] << " "
	     << network.predict(inps[1])[0] << " "
	     << network.predict(inps[2])[0] << " "
	     << network.predict(inps[3])[0] << " ";
}
