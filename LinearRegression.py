import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, x, y):
        self.x_vals = x
        self.y_vals = y
        self.theta0 = 0
        self.theta1 = 1
        self.__mean__ = sum(x)/len(x)
        self.__std__ = sum(x**2)/len(x) - (sum(x)/len(x))**2

    
    def gradientDescent(self,learingRate,numOfiterations):
        theta0 = self.theta0
        theta1 = self.theta1
        for i in range(numOfiterations):
            temp0 = theta0 - learingRate*sum((theta0 + theta1*self.x_vals - self.y_vals))/len(self.x_vals)
            temp1 = theta1 - learingRate*sum((theta0 + theta1*self.x_vals - self.y_vals)*self.x_vals)/len(self.x_vals)
            theta0 = temp0
            theta1 = temp1 
        self.theta0 = theta0
        self.theta1 = theta1
    
    def normalize(self,param):
        if(param == self.x_vals):
            mean = sum(self.x_vals)/len(self.y_vals)
            var = sum(self.x_vals**2)/len(self.x_vals) - mean**2
            std = var**(1/2)
            for i in range(len(self.x_vals)):
                self.x_vals[i] = (self.x_vals[i]-mean)/std
                
        elif(param == self.y_vals):
            mean = sum(self.y_vals)/len(self.y_vals)
            var = sum(self.y_vals**2)/len(self.y_vals) - mean**2
            std = var**(1/2)
            for i in range(len(self.x_vals)):
                self.y_vals[i] = (self.y_vals[i]-mean)/std

    def predict(self,x):
        return (self.theta0 + x*self.theta1)
        
    