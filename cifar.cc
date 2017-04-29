#include <iostream>
#include <fstream>
#include <string>

#include "Net.h"
#include "Layers.h"

using namespace std;

float **train = new float*[50000];
int *labels = new int[50000];
float **valid = new float*[10000];
int *validLabels = new int[10000];

void getData(){
	string filename;
	ifstream fin;
	for(int i=0; i<50000; ++i){
		if(i<10000) valid[i] = new float[3072];
		train[i] = new float[3072];
	}
	for(int i=0; i<5; ++i){
		filename = "/home/simmeboiii/code/cpp/ConvNet/cifar-10/data_batch_";
		filename += to_string(i+1);
		filename += ".bin";
		fin.open(filename);
		if(!fin){
			cout << "Error opening: " << i+1 << '\n';
			return;
		}
		for(int j=0; !fin.eof(); ++j){
			if(j%3073 == 0) labels[10000*i+j/3073] = fin.get();
			else train[10000*i+j/3073][j%3073-1] = (float)fin.get();
		}
		fin.close();
	}
	fin.open("/home/simmeboiii/code/cpp/ConvNet/cifar-10/test_batch.bin");
	for(int i=0; !fin.eof(); ++i){
		if(i%3073 == 0) validLabels[i/3073] = fin.get();
		else valid[i/3073][i%3073-1] = fin.get();
	}
}

void normalize(){
	float mean[3072];
	for(int i=0; i<3072; ++i) mean[i] = 0;
	for(int i=0; i<50000; ++i){
		for(int j=0; j<3072; ++j) mean[j] += train[i][j];
	}
	for(int i=0; i<3072; ++i) mean[i] /= 50000;
	for(int i=0; i<50000; ++i){
		for(int j=0; j<3072; ++j){
			if(i<10000) valid[i][j] -= mean[j];
			train[i][j] -= mean[j];
		}
	}
}

void validate(Net &net){
	int correct = 0;
	for(int i=0; i<10000; ++i){
		if(net.predict(valid[i]) == validLabels[i]) ++correct;
	}
	cout << "Stats: " << (float)correct/100 << '\n';
}

int main(){
	cout << "Loading data\n";
	getData();
	cout << "Normalizing\n";
	normalize();
	cout << "Creating net\n";
	Net net(new SoftMax(10));
	net.push(new Conv(Dim{3, 32, 32}, 10, 3, 1));
	net.push(new BatchNorm(Dim{10, 32, 32}));
	net.push(new ReLU(Dim{10, 32, 32}));

	net.push(new Conv(Dim{10, 32, 32}, 10, 3, 1));
	net.push(new BatchNorm(Dim{10, 32, 32}));
	net.push(new ReLU(Dim{10, 32, 32}));

	// net.push(new Conv(Dim{16, 32, 32}, 16, 5, 2));
	net.push(new MaxPool(Dim{10, 32, 32}));

	net.push(new Conv(Dim{10, 16, 16}, 20, 3, 1));
	net.push(new BatchNorm(Dim{20, 16, 16}));
	net.push(new ReLU(Dim{20, 16, 16}));

	net.push(new Conv(Dim{20, 16, 16}, 20, 3, 1));
	net.push(new BatchNorm(Dim{20, 16, 16}));
	net.push(new ReLU(Dim{20, 16, 16}));

	net.push(new MaxPool(Dim{20, 16, 16}));

	net.push(new Conv(Dim{20, 8, 8}, 20, 3, 1));
	net.push(new BatchNorm(Dim{20, 8, 8}));
	net.push(new ReLU(Dim{20, 8, 8}));

	net.push(new Conv(Dim{20, 8, 8}, 20, 3, 1));
	net.push(new BatchNorm(Dim{20, 8, 8}));
	net.push(new ReLU(Dim{20, 8, 8}));

	net.push(new MaxPool(Dim{20, 8, 8}));

	net.push(new FullyConn(Dim{20, 4, 4}, 10));

	cout << "Starting training\n";
	net.train(train, labels, 50000);
	validate(net);

	for(int i=0; i<50000; ++i){
		if(i<10000) delete[] valid[i];
		delete[] train[i];
	}
	std::cout << "Deleting Train: " << train << '\n';
	delete[] train;
	std::cout << "Done\n";
	std::cout << "Deleting Labels: " << labels << '\n';
	delete[] labels;
	std::cout << "Done\n";
	std::cout << "Deleting Valid: " << valid << '\n';
	delete[] valid;
	std::cout << "Done\n";
	std::cout << "Deleting ValidLabels: " << validLabels << '\n';
	delete[] validLabels;
	std::cout << "Done\n";
}