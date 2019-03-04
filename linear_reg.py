#This module contains a program that perforns linear regression on 
#two dimensional data --- A very smooth introduction to Machine Learning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Linear_Regression():
    
    def __init__(self, data):
        #data is a list of two list : th x-list and the y-list
        self.x_list = np.array(data[0])
        self.y_list = np.array(data[1])
        self.a = 0
        self.b = 0
        
    def normalise_data(self):
        
        #In case, we have different size lists
        min_size = min(len(self.x_list), len(self.y_list))
        self.x_list = self.x_list[:min_size]
        self.y_list = self.y_list[:min_size]
    
    def train_data(self):
        
        #The training part requires us to find the regression line, ie: its 
        #equation y  = ax + b
        #a = r sy/sx
        sy = np.std(self.y_list)
        sx = np.std(self.x_list)
        r = np.corrcoef(self.x_list, self.y_list)[0,1]  #np.corrcoef returns a matrix
        x_mean = np.mean(self.x_list)
        y_mean = np.mean(self.y_list)
        

        #-----YThe regression line is obtained
        self.b = r * sy/sx                          #The slope
        self.a = y_mean - self.b * x_mean                #The y-intercept
        
        print("The line of best fit is y = {:^4.3f}x + {:4^.3f}".format(self.b,self.a))
        
        return self.b, self.a
    
    def plot_result(self):
        plt.scatter(self.x_list, self.y_list, color = 'black')
        
        x = np.linspace(0,max(self.x_list), 10000)
        y = self.b * x + self.a
        
        plt.plot(x,y, color = 'green')
        
        
        plt.show()
        
        
        
def main():
    
    df = pd.read_excel('slr05.xls', encoding = 'ascii')
    
    data = [df['X'], df['Y']]
    
    l_r = Linear_Regression(data)
    l_r.normalise_data()
    l_r.train_data()
    l_r.plot_result()

if __name__ == "__main__":
    main()
        