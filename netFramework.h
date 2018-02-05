#pragma once
#include<iostream>
#include<cmath>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<vector>
#include<time.h>
#include<stdio.h>
#include"dataStorage.h"

using namespace std;

const double learningRate = 0.1, momentumConstant = 0, errorEnergyThreshold = 0.001; //eta, alpha, and tau respectively
const double sigmoidParameter = 0.01;
const double stabilityConstant = pow(errorEnergyThreshold, 2); //stopping condition for difference between trials
const int minTrialCount = 1000000; //bare minimum number of trials

double sigmoid(double value);
double randomDouble(double fMin, double fMax);
int randomInt(int fMin, int fMax);

/*contains pre- and post- activation function layers, and delta vector*/
struct neuronLayer
{
	neuronLayer();
	//default constructor
	neuronLayer(int layerSize);
	//constructs a layer with layerSize number of neurons
	void activation();
	//puts preActivation through the activation function to set postActivation
	void setErrorVector();
	//initializes error vector for output layer
	double *preActivation, *postActivation, *deltaVector;
	double *errorVector;
	int size; //number of neurons in layer
};

/* contains weight matrix and pointers from preceding neuronLayer and to next neuronLayer
has weight adjustment algorithm */
struct weightLayer
{
	weightLayer();
	//default constructor
	weightLayer(neuronLayer next, neuronLayer last);
	//constructor that connects weightLayer to previous layers and initializes weights
	void weightInitializer(int number);
	//initializes weights between (-1/sqrt(previousLayerSize), 1/sqrt(previousLayerSize))
	double **weightMatrix;
	double **deltaWeightMatrix;
	double **deltaWeightMatrixPrevious;
	int nextLayerSize, lastLayerSize;
};

// holds the structure of the net and initializes forward and back propagation
class neuralNet
{
public:
	neuralNet(string fileName);
	//constructs standard staight nets
	void beginTraining(dataStorage data);
	//initializes training sequence
	void beginTesting();
	//intitializes testing sequence
	int getTotalLayers();
	//returns totalLayers
	double getErrorTotal();
	//returns errorTotal
	void layerToFile(int fileNumber);
	//outputs weightlayers to files
	neuronLayer *neuronMap;
	weightLayer *weightMap;


private:
	int totalLayers;
	int *sizeOfLayers;
	void forwardPropagation(int index, dataStorage data); //starts forward propagation from index of data
	void backPropagation(int index, dataStorage data); //starts back Propagation from index of Labels
	void errorCalculation(); 
	double errorLast, errorNow, errorTotal; //error from previous trial, current, and total
	int trialCounter;
	bool stabilityCheck();//returns true if stopping conditions havent been met
};


neuralNet::neuralNet(string fileName)
{
	ifstream incomingData;
	incomingData.open(fileName);

	if (incomingData.fail())
	{
		cout << "Error: Could not open file" << endl;
		char dummy;
		cout << "Enter any key to quit." << endl;
		cin >> dummy;
		exit(1);
	}

	int dummy;
	incomingData >> dummy;
	totalLayers = dummy;

	sizeOfLayers = new int[totalLayers];

	for(int i = 0; i < totalLayers; i++)
	{
		incomingData >> sizeOfLayers[i];
	}
	incomingData.close();

	neuronMap = new neuronLayer[totalLayers];

	//fills in neuronMap and constructs neuronLayers
	for (int i = 0; i < totalLayers; i++)
	{
		neuronMap[i] = neuronLayer(sizeOfLayers[i]);
	}

	//initialize error vector for output layer
	neuronMap[totalLayers - 1].setErrorVector();

	weightMap = new weightLayer[totalLayers - 1];

	//fills in weightMap and constructs weightLayers and deltaWeightLayers
	for (int i = 0; i < totalLayers - 1; i++)
	{
		weightMap[i] = weightLayer(neuronMap[i + 1], neuronMap[i]);
	}
}

void neuralNet::beginTraining(dataStorage data)
{
	//initializes random initial weights
	for (int i = 0; i < totalLayers - 1; i++)
	{
		weightMap[i].weightInitializer(neuronMap[i].size);
	}

	int fileNumberCounter = 0;
	trialCounter = 1;
	do
	{
		errorLast = errorNow;
		int index = randomInt(0, data.getSampleSize() - 1);
		forwardPropagation(index, data);
		backPropagation(index, data);
		errorCalculation();
		if (trialCounter % 10 == 0)
		{
			cout << errorTotal << endl;
		}
		if (trialCounter % 10000 == 0)
		{
			layerToFile(fileNumberCounter);
			fileNumberCounter++;
		}
		trialCounter++;
	} while (true);
	cout << 'done' << endl;
}

inline void neuralNet::beginTesting()
{
	return; //not written
}

inline int neuralNet::getTotalLayers()
{
	return totalLayers;
}

inline double neuralNet::getErrorTotal()
{
	return errorTotal;
}

inline void neuralNet::layerToFile(int fileNumber)
{
	string matrixMap, matrix, number;
	ostringstream convert;
	convert << fileNumber;
	number = convert.str();
	matrixMap = "matrixMap" + number;
	matrixMap += ".txt";
	matrix = "matrix" + number;
	matrix += ".txt";

	ofstream outgoing;
	outgoing.open(matrixMap);
	if (outgoing.fail())
	{
		cout << "Error: Could not open file " << matrixMap << endl;
		char dummy;
		cout << "Enter any key to quit." << endl;
		cin >> dummy;
		exit(1);
	}

	outgoing << getErrorTotal() << endl;
	outgoing << trialCounter << endl;
	outgoing << totalLayers - 1 << endl;

	for (int i = 0; i < totalLayers - 1; i++)
	{
		outgoing << weightMap[i].lastLayerSize << ' ' << weightMap[i].nextLayerSize << endl;
	}
	outgoing.close();

	ofstream outgoing2;
	outgoing2.open(matrix);
	if (outgoing2.fail())
	{
		cout << "Error: Could not open file " << matrix << endl;
		char dummy;
		cout << "Enter any key to quit." << endl;
		cin >> dummy;
		exit(1);
	}
	for (int i = 0; i < totalLayers - 1; i++)
	{
		for (int j = 0; j < weightMap[i].lastLayerSize; j++)
		{
			for (int k = 0; k < weightMap[i].nextLayerSize; k++)
			{
				outgoing2 << weightMap[i].weightMatrix[j][k] << ' ';
			}
			outgoing2 << endl;
		}
		outgoing2 << endl;
	}
	return;
}


inline bool neuralNet::stabilityCheck()
{
	if (errorNow < errorEnergyThreshold && trialCounter > minTrialCount)
	{
		return 0;
	}
	else if(abs(errorNow - errorLast) < stabilityConstant && trialCounter > minTrialCount)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

inline void neuralNet::forwardPropagation(int index, dataStorage data)
{
	//initializes the input layer from data
	for (int i = 0; i < 28*28; i++)
	{
			neuronMap[0].preActivation[i] = static_cast<double>(data.data[index][i]);
	}
	neuronMap[0].activation();

	//does forward propagation
	double tempSum;
	for (int i = 1; i < totalLayers; i++)
	{
		for (int k = 0; k < neuronMap[i].size; k++)
		{
			tempSum = 0;
			for (int j = 0; j < neuronMap[i-1].size; j++)
			{
				tempSum += neuronMap[i - 1].postActivation[j] * weightMap[i - 1].weightMatrix[j][k];
			}
			neuronMap[i].preActivation[k] = tempSum;
		}
		neuronMap[i].activation();
	}
}

inline void neuralNet::backPropagation(int index, dataStorage data)
{
	//calculates error vector
	for (int i = 0; i < neuronMap[totalLayers - 1].size; i++)
	{
		if (i == data.labels[index])
		{
			neuronMap[totalLayers - 1].errorVector[i] = 1 - neuronMap[totalLayers - 1].postActivation[i];
		}
		else
		{
			neuronMap[totalLayers - 1].errorVector[i] = 0 - neuronMap[totalLayers - 1].postActivation[i];
		}
	}

	//calculates outputLayer's deltaVector
	for (int i = 0; i < neuronMap[totalLayers - 1].size; i++)
	{
		neuronMap[totalLayers -1].deltaVector[i] = neuronMap[totalLayers-1].errorVector[i] * sigmoidParameter * (neuronMap[totalLayers-1].postActivation[i] * (1 - neuronMap[totalLayers - 1].postActivation[i]));
	}

	//calculates outputLayer's deltaWeightMatrices and adjusts weightMatrix
	for (int i = 0; i < neuronMap[totalLayers - 2].size; i++)
	{
		for (int j = 0; j < neuronMap[totalLayers - 1].size; j++)
		{
			if (trialCounter == 1)
			{
				weightMap[totalLayers - 2].deltaWeightMatrix[i][j] = learningRate * neuronMap[totalLayers - 2].postActivation[i] * neuronMap[totalLayers - 1].deltaVector[j];
				weightMap[totalLayers - 2].deltaWeightMatrixPrevious[i][j] = weightMap[totalLayers - 2].deltaWeightMatrix[i][j];
				weightMap[totalLayers - 2].weightMatrix[i][j] += weightMap[totalLayers - 2].deltaWeightMatrix[i][j];
			}
			else
			{
				weightMap[totalLayers - 2].deltaWeightMatrix[i][j] = (learningRate * neuronMap[totalLayers - 2].postActivation[i] * neuronMap[totalLayers - 1].deltaVector[j]) + (momentumConstant * weightMap[totalLayers - 2].deltaWeightMatrixPrevious[i][j]);
				weightMap[totalLayers - 2].deltaWeightMatrixPrevious[i][j] = weightMap[totalLayers - 2].deltaWeightMatrix[i][j];
				weightMap[totalLayers - 2].weightMatrix[i][j] += weightMap[totalLayers - 2].deltaWeightMatrix[i][j];
			}
		}
	}

	
	//does the above for the rest of the layers
	double tempSum;
	for (int i = totalLayers - 2; i > 0; i--)
	{
		
		//calculates deltaVectors
		for (int j = 0; j < neuronMap[i].size; j++)
		{
			tempSum = 0;
			for (int k = 0; k < neuronMap[i + 1].size; k++)
			{
				tempSum += weightMap[i].weightMatrix[j][k] * neuronMap[i + 1].deltaVector[k];
			}
			neuronMap[i].deltaVector[j] = tempSum;
		}

		//calculates deltaWeightMatrices and adjusts
		for (int j = 0; j < neuronMap[i - 1].size; j++)
		{
			for (int k = 0; k < neuronMap[i].size; k++)
			{
				if (trialCounter == 1)
				{
					weightMap[i - 1].deltaWeightMatrix[j][k] = learningRate * neuronMap[i-1].postActivation[j] * neuronMap[i].deltaVector[k];
					weightMap[i - 1].deltaWeightMatrixPrevious[j][k] = weightMap[i - 1].deltaWeightMatrix[j][k];
					weightMap[i - 1].weightMatrix[j][k] += weightMap[i - 1].deltaWeightMatrix[j][k];
				}
				else
				{
					weightMap[i - 1].deltaWeightMatrix[j][k] = (learningRate * neuronMap[i - 1].postActivation[j] * neuronMap[i].deltaVector[k]) + (momentumConstant * weightMap[i - 1].deltaWeightMatrixPrevious[j][k]);
					weightMap[i - 1].deltaWeightMatrixPrevious[j][k] = weightMap[i - 1].deltaWeightMatrix[j][k];
					weightMap[i - 1].weightMatrix[j][k] += weightMap[i - 1].deltaWeightMatrix[j][k];
				}
			}
		}
	}
}

inline void neuralNet::errorCalculation()
{
	// Calculates the instantaneous sum of squared errors
	double tempSum = 0.0;
	for (int i = 0; i < neuronMap[totalLayers - 1].size; i++)
	{
		tempSum = tempSum + pow(neuronMap[totalLayers - 1].errorVector[i], 2);
	}
	errorNow = 0.5*tempSum;

	// Calculates average squared error
	if (trialCounter == 1)
	{
		errorTotal = errorNow;
	}
	else
	{
		errorTotal = errorTotal * (trialCounter - 1);
		errorTotal += errorNow;
		errorTotal = (1 / static_cast<double>(trialCounter))* errorTotal;
	}
}

inline weightLayer::weightLayer()
{
	nextLayerSize = 1;
	lastLayerSize = 1;
	weightMatrix = new double*[1];
	weightMatrix[0] = new double[1];
	deltaWeightMatrix = new double*[1];
	deltaWeightMatrix[0] = new double[1];
	deltaWeightMatrixPrevious = new double*[1];
	deltaWeightMatrixPrevious[0] = new double[1];
}

weightLayer::weightLayer(neuronLayer next, neuronLayer last)
{
	nextLayerSize = next.size;
	lastLayerSize = last.size;

	weightMatrix = new double*[lastLayerSize];
	deltaWeightMatrix = new double*[lastLayerSize];
	deltaWeightMatrixPrevious = new double*[lastLayerSize];
	for (int i = 0; i < lastLayerSize; i++)
	{
		weightMatrix[i] = new double[nextLayerSize];
		deltaWeightMatrix[i] = new double[nextLayerSize];
		deltaWeightMatrixPrevious[i] = new double[nextLayerSize];
	}
}

inline void weightLayer::weightInitializer(int number)
{
	double min, max;
	min = -1 / (sqrt(static_cast<double>(number)));
	max = 1 / (sqrt(static_cast<double>(number)));
	for (int i = 0; i < lastLayerSize; i++)
	{
		for (int j = 0; j < nextLayerSize; j++)
		{
			weightMatrix[i][j] = randomDouble(min, max);
		}
	}
}

inline neuronLayer::neuronLayer()
{
	size = 1;
	preActivation = new double[1];
	postActivation = new double[1];
	deltaVector = new double[1];
}

neuronLayer::neuronLayer(int layerSize)
{
	size = layerSize;
	preActivation = new double[layerSize];
	postActivation = new double[layerSize];
	deltaVector = new double[layerSize];
}

inline void neuronLayer::activation()
{
	for (int i = 0; i < size; i++)
	{
		postActivation[i] = sigmoid(preActivation[i]);
	}
}

inline void neuronLayer::setErrorVector()
{
	errorVector = new double[size];
}

double randomDouble(double fMin, double fMax)
{
	double f = static_cast<double>(rand()) / RAND_MAX;
	return fMin + f * abs(fMax - fMin);
}

int randomInt(int fMin, int fMax)
{
	int rangeAdjustment;
	int producedRand;

	rangeAdjustment = abs(fMax - fMin);

	if (rangeAdjustment <= RAND_MAX)
	{
		int remainder = RAND_MAX % (rangeAdjustment + 1);
		do {
			producedRand = rand();
		} while (producedRand < remainder);
		return fMin + (producedRand % (rangeAdjustment + 1));
	}
	else
	{
		do {
			producedRand = (rand())*((rangeAdjustment / RAND_MAX) + 1) + randomInt(0, (rangeAdjustment / RAND_MAX));
		} while (producedRand > rangeAdjustment);
	}
	return (producedRand + fMin);
}

double sigmoid(double value)
{
	double newNumber;
	newNumber = (1 / (1 + exp((-1 * sigmoidParameter * value)))) - 0.5;
	return newNumber;
}