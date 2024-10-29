import numpy as np
import find_filter_coefficients as find_fc
import find_wavelet_coefficients as fwc
import matplotlib.pyplot as plt
import load_data as ld
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def fitness_function_grid_search(X,y):
    #initialize vector to store fitness values
    fitness_values = []
    fitness_function_details=[]

    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    # Initialize the GridSearchCV
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')

    for i in range(X.shape[0]):
        # Fit the grid search to the data
        grid_search.fit(X[i,:,:], y) 

        # Get the best parameters and best score  
        particle_details=grid_search.best_params_
        particle_details["fitness"]=grid_search.best_score_
        
        fitness_values.append(grid_search.best_score_)
        fitness_function_details.append(particle_details)

    return fitness_values, fitness_function_details

def fitness_function_CV(X,y):
    #initialize vector to store fitness values
    fitness_values = []


    # Initialize fixed SVC model parameters
    model = SVC(kernel='rbf', C=100, gamma='scale')  # Set C and gamma as desired

    for i in range(X.shape[0]):
        # Fit the grid search to the data
        scores = cross_val_score(model, X[i, :, :], y, cv=5, scoring='accuracy')

        # Get the best parameters and best score  
        fitness_values.append(np.mean(scores))
        

    fitness_values = np.array(fitness_values)
    print("fitness value is", np.mean(fitness_values) )
    return fitness_values