#ifndef NET_H
#define NET_H

#include <random>

#include "Layers.h"

class Net{
	Layer **layers = nullptr;
	Loss *lossFunc = nullptr;
	int depth = 0;
	int batchSize = 50;
	int epochs = 10;
	float learnRate = 0.001;
	float regStrength = 0.00001;

public:

	Net(Loss *loss): lossFunc(loss){}

	void push(Layer *layer){
		if(depth == 0){
			layers = new Layer*;
			*layers = layer;
			++depth;
		}
		else{
			Layer **temp = layers;
			layers = new Layer*[depth+1];
			for(int i=0; i<depth; ++i) layers[i] = temp[i];
			layers[depth++] = layer;
			delete[] temp;
		}
	}

	int predict(float *data){
		int guess;
		float max;
		float *temp = new float[3072];
		for(int i=0; i<3072; ++i) temp[i] = data[i];
		for(int i=0; i<depth; ++i) temp = layers[i]->forward(temp);
		max = temp[0];
		guess = 0;
		for(int i=1; i<10; ++i){
			if(temp[i] > max){
				guess = i;
				max = temp[i];
			}
		}
		delete[] temp;
		return guess;
	}

	void train(float **data, int *labels, int num_examples){
		std::random_device rd;
		std::mt19937 rand(rd());
		float loss = 0;
		float *current = new float[3072];
		for(int i=0, index; i<num_examples*epochs; ++i){
			index = rand()%num_examples;

			for(int j=0; j<3072; ++j) current[j] = data[index][j];
			lossFunc->setCorrect(labels[index]);

			for(int j=0; j<depth; ++j) current = layers[j]->forward(current);
			current = lossFunc->forward(current);

			loss += lossFunc->getLoss();

			current = lossFunc->backward(current);
			for(int j=depth-1; j>=0; --j) current = layers[j]->backward(current);

			if((i+1)%batchSize == 0){
				for(int i=0; i<depth; ++i) layers[i]->update(learnRate, regStrength);
				std::cout << (i+1)/batchSize << " : " << i+1 << 
						"\nBatch Loss: " << (float)loss/batchSize << "\n\n";
				loss = 0;
			}
		}
	}

	~Net(){
		delete[] layers;
		delete[] lossFunc;
	}
};

#endif