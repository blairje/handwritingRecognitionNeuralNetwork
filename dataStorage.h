#pragma once
#include<iostream>
#include<cmath>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<vector>
#include<time.h>
#include<stdio.h>

using namespace std;

class dataStorage
{
public:
	dataStorage(int size, string labelFile, string dataFile);
	//constructs trainingData for "size" number of samples
	int *labels; //1D dynamic array storing labels for trainingData, values 0-9
	int **data; //3D dynamic array [image#][Pixel]
	int getSampleSize(); //returns sampleSize
private:
	int sampleSize; //number of samples
	void fileInput(int size, string fileName, string fileName2); //inputs file into trainingLabels and trainingData
};

dataStorage::dataStorage(int size, string labelFile, string dataFile)
{
	sampleSize = size;
	labels = new int[size];
	data = new int*[size];
	for (int i = 0; i < size; i++)
	{
		data[i] = new int[28*28];
	}
	fileInput(size, labelFile, dataFile);
}

inline int dataStorage::getSampleSize()
{
	return sampleSize;
}

inline void dataStorage::fileInput(int size, string fileName, string fileName2)
{

	int counter = 0;
	char * junk;
	junk = new char[size];
	std::ifstream file(fileName, std::ios::binary);
	if (file.is_open())
	{
		file.seekg(8, std::ios::beg);
		file.read(junk, size);
		file.close();

		for (int i = 0; i < size; i++)
		{
			labels[i] = static_cast<int>(junk[i]);
		}

		delete[] junk;
	}
	else std::cout << "Unable to open file:" << fileName;

	counter = 0;
	int pixelCount = size * 28 * 28;
	junk = new char[pixelCount];
	std::ifstream file2(fileName2, std::ios::binary);
	if (file2.is_open())
	{
		file2.seekg(16, std::ios::beg);
		file2.read(junk, pixelCount);
		file2.close();

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < 28*28; j++)
			{
				data[i][j] = static_cast<int>(junk[counter]);
				counter++;
			}
		}
		delete[] junk;
	}
	else std::cout << "Unable to open file:" << fileName2;
	return;
}
