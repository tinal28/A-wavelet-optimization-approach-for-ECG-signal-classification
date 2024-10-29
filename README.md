# Wavelet Optimization for ECG Signal Classification
Generating the wavelet that best represents the ECG beats in terms of discrimination capability 

## Overview

This project implements a wavelet optimization approach for ECG signal classification using Particle Swarm Optimization (PSO). The goal is to enhance the accuracy of classification by optimizing the parameters of wavelet transformations applied to the ECG signals.

## Requirements

Make sure you have Python installed along with the necessary libraries. You can install the required libraries using:

## Functionality
Data Loading: Load ECG signal data from the MIT-BIH database.
Data Preprocessing: Split the dataset into training and testing sets.
Create custom wavelets using sherlock Monro Algorithm 
Wavelet Transformation: Apply wavelet decomposition to the ECG signals.
Optimize the parameters of wavelet transformation using Particle Swarm Optimization (PSO).
Classification: Evaluate the classification performance using k-fold cross validation using SVM.

## Results
After 20 iterations the generated mother wavelet is displayed below.
![mother wavelet for best accuracy](https://github.com/user-attachments/assets/01893bd9-da62-40da-9d39-19a5897f62b1)

## References
Abdelhamid Daamouche, Latifa Hamami, Naif Alajlan, Farid Melgani,
A wavelet optimization approach for ECG signal classification,
Biomedical Signal Processing and Control, 
Volume 7, Issue 4,
2012,
Pages 342-349,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2011.07.001.

(https://www.sciencedirect.com/science/article/pii/S1746809411000772)

Abstract: Wavelets have proved particularly effective for extracting discriminative features in ECG signal classification. In this paper, we show that wavelet performances in terms of classification accuracy can be pushed further by customizing them for the considered classification task. A novel approach for generating the wavelet that best represents the ECG beats in terms of discrimination capability is proposed. It makes use of the polyphase representation of the wavelet filter bank and formulates the design problem within a particle swarm optimization (PSO) framework. Experimental results conducted on the benchmark MIT/BIH arrhythmia database with the state-of-the-art support vector machine (SVM) classifier confirm the superiority in terms of classification accuracy and stability of the proposed method over standard wavelets (i.e., Daubechies and Symlet wavelets).

Keywords: Classification; Discrete wavelet transform (DWT); Electrocardiogram (ECG) signals; Particle swarm optimization (PSO); Support vector machines (SVM)
