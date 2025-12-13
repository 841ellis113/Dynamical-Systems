import numpy as np
import matplotlib.pyplot as plt
import random 
import seaborn as sns
import pandas as pd

'''
Using Pandas to investigate the Logistic Map
'''

'''
calculating values for a number of iterations
'''
def logistic_map_values(r,x0,iterations,transients):
    x = x0 + random.uniform(0,1)*1e-5
    x_values   = []
    for i in range(iterations):
        x = r*x*(1-x)
        if i > transients:
            x_values.append(x)
    return x_values

'''
creates a dataframe using pandas of the lostic map values for a range of r and dispalys in
wide format (1 column per r value) as well as the running average of 5 iterations
'''

def logistic_dataframe(R_list,x0,iterations,transients,shape='tall'):
    storage = []
    for r in R_list:
        x_values  = logistic_map_values(r,x0,iterations,transients)
        df        = pd.DataFrame({'Iterations':range(len(x_values)),'r':r,'X':x_values})
        storage.append(df)
    dataframe = pd.concat(storage)
    if shape == 'wide':
        dataframe = dataframe.pivot(index='Iterations',columns='r',values='X')      
    return dataframe

'''
create a database that contains an embedded embedded matrix of the Logistic Map for a
specific value of the parameter r. Inputs ar a list of time delays, t and must be integer.
'''

def embedded_dataframe(t_list,r,x0,iterations,transients,shape='tall'):
    storage  = []
    stop     = max(t_list)+1
    x_values = logistic_map_values(r,x0,iterations,transients)
    t_list.insert(0,0)
    for t in t_list:
        df   = pd.DataFrame({
             'Iterations':range(len(x_values)-stop),
             't':t,
             'X':x_values[t:-stop+t]
             })
        storage.append(df)
    dataframe = pd.concat(storage)
    if shape == 'wide':
        dataframe = dataframe.pivot(index='Iterations',columns='t',values='X')
    return dataframe

'''
take a dataframe as an input and returns a datarame that also contains the derivative and
instantaneous Lyapunov exponent for each value of X for each r. It also returns the
Lyapunov exponent (mean) for each parameter value as a seperate array.
Input must be of teh 'tall' type or rather the returned form from the function
logistic_dataframe. The output can be either the wide or tall type but wide will have
multiindexing.
'''

def Lyapunov_dataframe(DF,out_put='tall'):
    DF["X'"] = DF.apply(lambda row: row['r']*(1-2*row['X']), axis=1)
    DF['Lyapunov'] = DF["X'"].apply(lambda value: np.log(np.abs(value)))
    exponent = DF.groupby('r')['Lyapunov'].mean()
    if output == 'tall':
        dataframe = DF
    elif output == 'wide':
        DF = DF.pivot(index='Iterations',columns='r',values=["X","X'","Lyapunov"])
        DF = DF.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        dataframe = DF
    return Dataframe, exponent
