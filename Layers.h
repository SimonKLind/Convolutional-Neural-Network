#ifndef LAYERS_H
#define LAYERS_H

#include <cmath>
#include <random>

struct Dim{
	int z;
	int y;
	int x;
};

class Layer{
protected:
	float *latestIn = nullptr;
	Dim in;
	int inLength;
public:
	Layer(Dim inDim): in(inDim), inLength(inDim.x*inDim.y*inDim.z){}
	virtual float* forward(float *input) = 0;
	virtual float* backward(float *grads) = 0;
	virtual void update(float learnRate, float regStrength){}

	virtual ~Layer(){
		delete[] latestIn;
	}
};

class Conv: public Layer{
	float *filters;
	float *gradients;
	float *cache;
	float *biases;
	float *biasGrads;
	int num, size, stride, pad, outSize;
	int fLength;

public:

	Conv(Dim in, int num_filters, int filter_size, int stride_count): 
	Layer(in), num(num_filters), size(filter_size), stride(stride_count){
		std::random_device rd;
		std::mt19937 rand(rd());
		std::normal_distribution<float> dist(0.0, 1.0);
		pad = filter_size>>1;
		outSize = in.x/stride_count;
		fLength = filter_size*filter_size*in.z*num_filters;
		filters = new float[fLength];
		gradients = new float[fLength];
		cache = new float[fLength];
		biases = new float[num_filters];
		biasGrads = new float[num_filters];
		float rootN = sqrt((float)2/(in.x*in.y*in.z));
		for(int i=0; i<fLength; ++i){
			if(i<num_filters) biases[i] = biasGrads[i] = 0;
			filters[i] = dist(rand)*rootN;
			gradients[i] = 0;
			cache[i] = 0;
		}
	}

	float* forward(float *input){
		delete[] latestIn;
		latestIn = input;
		input = new float[outSize*outSize*num];
		int oy, ox, filterIndex, outIndex, inIndex;
		for(int filter=0; filter<num; ++filter){
			for(int y=0; y<outSize; ++y){
				oy = y*stride-pad;
				for(int x=0; x<outSize; ++x){
					ox = x*stride-pad;
					outIndex = (filter*outSize+y)*outSize+x;
					input[outIndex] = 0;
					for(int fy=0; fy<size; ++fy){
						for(int fx=0; fx<size; ++fx){
							if(fx+ox >= 0 && fx+ox < in.x && fy+oy >= 0 && fy+oy < in.y){
								for(int fz=0; fz<in.z; ++fz){
									filterIndex = ((filter*in.z+fz)*size+fy)*size+fx;
									inIndex = (fz*in.y+fy+oy)*in.x+fx+ox;
									input[outIndex] += filters[filterIndex]*latestIn[inIndex];
								}
							}
						}
					}
					input[outIndex] += biases[filter];
				}
			}
		}
		return input;
	}

	float* backward(float *grads){
		float *global = grads;
		grads = new float[inLength];
		for(int i=0; i<inLength; ++i) grads[i] = 0;
		int oy, ox, filterIndex, outIndex, inIndex;
		for(int filter=0; filter<num; ++filter){
			for(int y=0; y<outSize; ++y){
				oy = y*stride-pad;
				for(int x=0; x<outSize; ++x){
					ox = x*stride-pad;
					outIndex = (filter*outSize+y)*outSize+x;
					for(int fy=0; fy<size; ++fy){
						for(int fx=0; fx<size; ++fx){
							if(fx+ox >= 0 && fx+ox < in.x && fy+oy >= 0 && fy+oy < in.y){
								for(int fz=0; fz<in.z; ++fz){
									filterIndex = ((filter*in.z+fz)*size+fy)*size+fx;
									inIndex = (fz*in.y+fy+oy)*in.x+fx+ox;
									grads[inIndex] += global[outIndex]*filters[filterIndex];
									gradients[filterIndex] += global[outIndex]*latestIn[inIndex];
								}
							}
						}
					}
					biasGrads[filter] += global[outIndex];
				}
			}
		}
		delete[] global;
		return grads;
	}

	void update(float learnRate, float regStrength){
		for(int i=0; i<fLength; ++i){
			if(i<num){
				biases[i] -= biasGrads[i]*learnRate;
				biasGrads[i] = 0;
			}
			gradients[i] += regStrength*filters[i];
			cache[i] = 0.99*cache[i] + 0.01*gradients[i]*gradients[i];
			filters[i] -= learnRate*gradients[i]/(sqrt(cache[i]) + 0.00000001);
			gradients[i] = 0;
		}
	}

	~Conv(){
		delete[] filters;
		delete[] gradients;
		delete[] cache;
		delete[] biases;
		delete[] biasGrads;
	}
};

class FullyConn: public Layer{
	float *neurons;
	float *gradients;
	float *cache;
	float *biases;
	float *biasGrads;
	int num, nLength;

public:

	FullyConn(Dim in, int num_neurons): Layer(in), num(num_neurons){
		std::random_device rd;
		std::mt19937 rand(rd());
		std::normal_distribution<float> dist(0.0, 1.0);
		nLength = num_neurons*in.x*in.y*in.z;
		neurons = new float[nLength];
		gradients = new float[nLength];
		cache = new float[nLength];
		biases = new float[num_neurons];
		biasGrads = new float[num_neurons];
		float rootN = sqrt((float)2/(in.x*in.y*in.z));
		for(int i=0; i<nLength; ++i){
			if(i<num_neurons) biases[i] = biasGrads[i] = 0;
			neurons[i] = dist(rand)*rootN;
			gradients[i] = 0;
			cache[i] = 0;
		}
	}

	float* forward(float *input){
		delete[] latestIn;
		latestIn = input;
		input = new float[num];
		for(int i=0; i<num; ++i){
			input[i] = 0;
			for(int j=0; j<inLength; ++j) input[i] += neurons[i*inLength+j]*latestIn[j];
			input[i] += biases[i];
		}
		return input;
	}

	float* backward(float *grads){
		float *global = grads;
		grads = new float[inLength];
		for(int i=0; i<inLength; ++i){
			grads[i] = 0;
			for(int j=0; j<num; ++j){
				gradients[j*inLength+i] += global[j]*latestIn[i];
				grads[i] += global[j]*neurons[j*inLength+i];
			}
		}
		delete[] global;
		return grads;
	}

	void update(float learnRate, float regStrength){
		for(int i=0; i<nLength; ++i){
			if(i<num){
				biases[i] -= learnRate*biasGrads[i];
				biasGrads[i] = 0;
			}
			gradients[i] += regStrength*neurons[i];
			cache[i] = 0.99*cache[i] + 0.01*gradients[i]*gradients[i];
			neurons[i] -= learnRate*gradients[i]/(sqrt(cache[i]) + 0.00000001);
			gradients[i] = 0;
		}
	}

	~FullyConn(){
		delete[] neurons;
		delete[] gradients;
		delete[] cache;
		delete[] biases;
		delete[] biasGrads;
	}
};

class BatchNorm: public Layer{
	float y, b;
	float dy, db;
	float mean, variance, ivar;
	float cache;
	float *out;

public:

	BatchNorm(Dim in): Layer(in){
		out = new float[in.x*in.y*in.z];
		y=1;
		dy=b=db=cache=0;
	}

	float* forward(float *input){
		delete[] latestIn;
		latestIn = input;
		input = new float[inLength];
		mean = 0;
		for(int i=0; i<inLength; ++i) mean += latestIn[i];
		mean /= inLength;
		variance = 0;
		for(int i=0; i<inLength; ++i) variance += (latestIn[i]-mean)*(latestIn[i]-mean);
		variance /= inLength;
		ivar = 1/sqrt(variance);
		for(int i=0; i<inLength; ++i){
			out[i] = (latestIn[i]-mean)*ivar;
			input[i] = y*out[i]+b;
		}
		return input;
	}

	float* backward(float *grads){
		float sumGlobal = 0, sumGxOut = 0;
		for(int i=0; i<inLength; ++i){
			sumGlobal += grads[i];
			sumGxOut += grads[i]*out[i];
		}
		float common = y*ivar/inLength;
		for(int i=0; i<inLength; ++i){
			dy += grads[i]*out[i];
			db += grads[i];
			grads[i] = common*(grads[i]*inLength - sumGlobal - out[i]*sumGxOut);
		}
		return grads;
	}

	void update(float learnRate, float regStrength){
		cache = 0.99*cache + 0.01*dy*dy;
		y -= learnRate*dy/(sqrt(cache) + 0.0000001);
		b -= learnRate*db;
		dy=db=0;
	}
};

class MaxPool: public Layer{
	int outSize;

	float* getMax(float *max, float *test){
		if(*max >= *test){
			*test = 0;
			return max;
		}
		else{
			*max = 0;
			return test;
		}
	}

public:

	MaxPool(Dim in): Layer(in), outSize(in.x>>1){}

	float* forward(float *input){
		delete[] latestIn;
		latestIn = input;
		input = new float[outSize*outSize*in.z];
		int ox, oy;
		float *max;
		for(int z=0; z<in.z; ++z){
			for(int y=0; y<outSize; ++y){
				oy=y<<1;
				for(int x=0; x<outSize; ++x){
					ox=x<<1;
					max = &latestIn[(z*in.x+oy)*in.y+ox];
					max = getMax(max, &latestIn[(z*in.x+oy)*in.y+ox+1]);
					max = getMax(max, &latestIn[(z*in.x+oy+1)*in.y+ox]);
					max = getMax(max, &latestIn[(z*in.x+oy+1)*in.y+ox+1]);
					input[(z*outSize+y)*outSize+x] = *max;
					*max = 1;
				}
			}
		}
		return input;
	}

	float* backward(float *grads){
		float *global = grads;
		grads = new float[inLength];
		int ox, oy;
		for(int z=0; z<in.z; ++z){
			for(int y=0; y<in.y; ++y){
				oy = y>>1;
				for(int x=0; x<in.x; ++x){
					ox = x>>1;
					grads[(z*in.x+y)*in.y+x] = latestIn[(z*in.x+y)*in.y+x]*global[(z*outSize+oy)*outSize+ox];
				}
			}
		}
		delete[] global;
		return grads;
	}
};

class ReLU: public Layer{
public:

	ReLU(Dim in): Layer(in) {
		latestIn = new float[in.x*in.y*in.z];
	}

	float* forward(float *input){
		for(int i=0; i<inLength; ++i){
			if(input[i] <= 0) latestIn[i] = input[i] = 0;
			else latestIn[i] = 1;
		}
		return input;
	}

	float* backward(float *grads){
		for(int i=0; i<inLength; ++i) grads[i] *= latestIn[i];
		return grads;
	}
};

class Loss: public Layer{

protected:
	int correct = -1;
	float loss;

public:

	Loss(int classes): Layer(Dim{1, 1, classes}) {
		latestIn = new float[in.x*in.y*in.z];
	}

	virtual float* forward(float *input) = 0;
	virtual float* backward(float *grads) = 0;

	void setCorrect(int index){
		correct = index;
	}

	float getLoss(){
		return loss;
	}
};

class SoftMax: public Loss{
	float total;

public: 

	SoftMax(int classes): Loss(classes){}

	float* forward(float *input){
		float max = input[0];
		for(int i=1; i<inLength; ++i){
			if(input[i] > max) max = input[i];
		}
		total = 0;
		for(int i=0; i<inLength; ++i){
			latestIn[i] = exp(input[i]-max);
			total += latestIn[i];
		}
		loss = -log(latestIn[correct]/total);
		return input;
	}

	float* backward(float *grads){
		for(int i=0; i<inLength; ++i) grads[i] = latestIn[i]/total;
		grads[correct] -= 1;
		return grads;
	}
};

#endif