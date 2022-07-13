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
const double BIAS_LOWER_BOUND = 2;

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

	vector<double> trainLayer(vector<double> de_dh, vector<double> prev_values) {
		for (int i = 0; i < num_of_neurons; i++) {
			double lr = -0.01;
			double de_dbi = de_dh[i]*df(neurons[i].prev_value);
			neurons[i].bias += lr*de_dbi;
			double de_dw;
			for (int j = 0; j < num_of_axons; j++) {
				de_dw = de_dh[i]*df(neurons[i].prev_value)*actFunc(prev_values[j]);
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

	double fit(const vector<double>& x, vector<double> y) {
		int size_of_ans = (int) y.size();
		vector<double> y_pred = predict(x);
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
		vector<double> prev_values(x.size());
		for (int i = num_of_layers - 1; i > 0; i--) {
			int prev_size = layers[i].num_of_axons;
			for(int j = 0; j < prev_size; j++){
				prev_values[j] = layers[i-1].neurons[j].prev_value;
			}
			de_dt = layers[i].trainLayer(de_dt, prev_values);
		}
		layers[0].trainLayer(de_dt, x);
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
	vector<int> amount = {4, 3, 5, 2};
	network.makeLayers(amount);
	int examples = 20;
	vector<double> inps[] = {{5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {5.1, 3.5, 1.4, 0.2},
	                         {6.3, 3.3, 6.0, 2.5},
	                         {5.8, 2.7, 5.1, 1.9},
	                         {7.1, 3.0, 5.9, 2.1},
	                         {6.3, 2.9, 5.6, 1.8},
	                         {6.5, 3.0, 5.8, 2.2},
	                         {7.6, 3.0, 6.6, 2.1},
	                         {4.9, 2.5, 4.5, 1.7},
	                         {7.3, 2.9, 6.3, 1.8},
	                         {6.7, 2.5, 5.8, 1.8},
	                         {7.2, 3.6, 6.1, 2.5},

	};
	vector<double> answs[] = {{0, 1}, //Iris - setosa
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {0, 1},
	                          {1, 0}, //Iris - virginica
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0},
	                          {1, 0}};

	double permissible_mse = 0.03;
	for (int i = 0; i < 250; i++) {
		double mse = 10;
		int j = 0;
		network.shuffle();
		while (mse > permissible_mse && j < 100) {
			mse = 0;
			for (int k = 0; k < examples*10; k++) {
				mse += network.fit(inps[k%examples], answs[k%examples]);
			}
			mse /= 10*examples;
			j++;
		}
		cout << "\nMSE: " << mse;
		if (mse <= permissible_mse) {
			cout<<"\nSuccess!\n";
			break;
		}
	}
	inps[0] = {5.0, 3.4, 1.6, 0.4}; // Iris - setosa
	inps[1] = {5.2, 3.5, 1.5, 0.2};
	inps[2] = {5.8, 2.7, 5.1, 1.9}; // Iris - virginica
	inps[3] = {6.8, 3.2, 5.9, 2.3};

	cout << "\nPredictions: \n" << network.predict(inps[0])[0] << " " << network.predict(inps[0])[1] << endl;
	cout << network.predict(inps[1])[0] << " " << network.predict(inps[1])[1] << endl;
	cout << network.predict(inps[2])[0] << " " << network.predict(inps[2])[1] << endl;
	cout << network.predict(inps[3])[0] << " " << network.predict(inps[3])[1] << endl;
}

