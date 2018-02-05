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
#include"netFramework.h"

using namespace std;


int main()
{
	srand(time(NULL));
	//the following code is for training
	dataStorage digitTrainingData = dataStorage(60000, "trainingLabels", "trainingImages");
	neuralNet digitNet = neuralNet("layerMap.txt");
	digitNet.beginTraining(digitTrainingData);
	


	return 0;
}