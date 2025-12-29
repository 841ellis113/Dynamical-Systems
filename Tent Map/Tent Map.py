import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

'''
First we map the tent function essentially for 1 time step and as a function of the
initial value of x. The tent map has the form of if x<0.5 then T(x) = rx and if
x>0.5 then T(x) = r(1-x) with the parameter being [0,2] and the domain [0,1]. The first
function is essentially 1 iteration within a time series.
'''

def Tent(r,x):
    if x <0.5:
        return r*x
    else:
        return r*(1-x)

'''
next we then output an array of values for the orbit or iterations of the tent map for
a given value of the parameter, r and initial coordinate, x0, for a specified number of
iterations and remove a number of transients to let any periodic behaviour to emerge. As the
Lyapunov exponent is just ln|r| at each x, we also calulate this for the given r. The
input of the function is the parameter, the initial value, the number of time steps and
the number of transients to remove. The removal of transients is automatic and the output
will be a pandas database that has iterations numbered from n=transients+1 and its values.
This is set by remove= True. If you would like to retain the transients in the output, the
remove argument must be set to False.
'''

def Tent_values(r,x,iterations,transients,remove = True):
    raw_steps    = []
    raw_x_values = []
    for i in range(iterations+1):
        raw_steps.append(i)
        raw_x_values.append(x)
        x = Tent(r,x)
    if remove == True:
        steps    = raw_steps[transients:]
        x_values = raw_x_values[transients:]
    else:
        steps    = raw_steps
        x_values = raw_x_values
    data = pd.DataFrame({'Iterations':steps,'r':r,'X_n':x_values, 'Lyapunov':np.log(r)})   
    return data
    
'''
Now we iterate the map for many values of r between 0 and 2, then plot the x values as a
function of r using the Tent_values() function. This will be the bifurcation map and we
also graph the Lyapunov exponent using Tent_Lyapunov(). Both need the Tent_values function
and produce graphs for a specified r range and number of r points in that range.
'''

def Tent_bifurcation(r1,r2,x,iterations,transients,spacing,remove = True,graph=True,ax=None):
    r_range = np.linspace(r1,r2,spacing)
    if ax == None:
        ax = plt.gca()
    for r in r_range:
        data = Tent_values(r,x,iterations,transients,remove=True)
        ax.scatter(data['r'],data['X_n'],color = 'black',s=0.3,alpha=0.1)
    if graph == True:
        plt.title('Tent Map Bifurcation')
        plt.xlabel('Parameter Value (r)')
        plt.xlim(r1,r2)
        plt.ylabel('Orbit Value (x)')
        plt.ylim(0,1)
        plt.grid(True,which='both',linestyle='--',color='grey',linewidth=0.7)
        plt.show()

def Tent_Lyapunov(r1,r2,x,iterations,transients,spacing,remove=True,graph=True,ax=None):
    r_range = np.linspace(r1,r2,spacing)
    if ax == None:
        ax = plt.gca()
    for r in r_range:
        data = Tent_values(r,x,iterations,transients,remove=True)
        ax.scatter(data['r'],data['Lyapunov'],color = 'red',s=0.4,alpha=0.5)
    if graph == True:
        plt.title('Tent Lyapunov Values')
        plt.xlabel('Parameter Value (r)')
        plt.grid(True, which='both', linestyle='--',color='grey',linewidth=0.8)
        plt.axhline(0,r1,r2,color='black',linewidth=1,linestyle='--')
        plt.axvline(1,-7,1,color='black',linewidth=1,linestyle='--')
        plt.xlim(r1,r2)
        plt.ylabel('λ')
        plt.ylim(-7,1)
        plt.show()

'''
using both the Tent_Lyapunov and Tent_bifurcation function, we can overlay the two on the
same graph
'''

def Tent_overlay_1(r1,r2,x,iterations,transients,spacing,remove=True,graph=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Tent_bifurcation(r1,r2,x,iterations,transients,spacing,remove=True,graph=False,ax=ax2)
    Tent_Lyapunov(r1,r2,x,iterations,transients,spacing,remove=True,graph=False,ax=ax1)
    ax1.set_xlim(0,2)
    ax1.set_ylim(-2,0.7)
    ax1.grid(True,which='both',linestyle='--',color='grey',linewidth=0.7)
    ax1.set_xlabel('Parameter (r)')
    ax2.set_ylabel('Orbit values (X)')
    ax2.set_ylim(0,1)
    ax1.set_ylabel('Lyapunov exponent')
    ax1.axhline(0,linestyle='--',color='black',linewidth=1)
    plt.suptitle('Comparing Lyapunov Exponent to Bifurcation')
    plt.show()

'''
Return maps are useful for looking at the geometry of the orbits of the map. We use a
dataframe created from Tent_values as a generator within the function and then we
append the number of columns for the number of time delay i.e t=1,2,3....
We then tidy up the dataframe using pandas method of dropna(). this results in the
dataframe being shorter in rows by t rows, where t is the number of time delayed columns.
The output is an embedded matrix which is used to produce return maps of differing time
delays
'''

def embedded_matrix(r,x,tau,iterations,transients,remove = True):
    data = Tent_values(r,x,iterations,transients,remove=True)
    for t in range(1,1+tau):
        column = []
        for L in range(len(data)):
            try:
                column.append(data['X_n'].iloc[L+t])
            except IndexError:
                column.append(None)
        data[f'X_n+{t}'] = column
    wip = data.pop('Lyapunov')
    data['Lyapunov'] = wip
    data = data.dropna()
    return data

def Tent_return_map(r,x,tau,iterations,transients,remove=True):
    data = embedded_matrix(r,x,tau,iterations,transients)
    fig,ax = plt.subplots(1,tau)
    for t in range(tau):
        ax[t].scatter(data[f'X_n'],data[f'X_n+{t+1}'],s=0.5,color='black',alpha=0.9)
        ax[t].grid(True, which='both', linestyle='--', color = 'grey', linewidth=0.5)
        ax[t].set_xlim(0,1)
        ax[t].set_ylabel(f'$X_{{n+{t+1}}}$')
        ax[t].set_ylim(0,1)
        ax[t].set_title(f'τ = {t+1}')
    fig.suptitle(f'Return Maps for Tent Map up to τ = {tau}')
    fig.supxlabel('$X_n$')
    plt.show()
            
'''
Calculating the invariant density by first removing transients, then dividing the domain [0,1]
into a number of spacings or bins and allocating each point into one of the bins. (Done by
multipling the data points by number of bins, then using floor function to allocate them.
Then we calculate the probability by dividing the total number in each bin by the total
number of data points and finally multiplying by bin width to get probability density.
'''

def Tent_Inv_density(r,x,iterations,transients,bins):
    width     = 1/bins
    data      = Tent_values(r,x,iterations,transients,remove = True)['X_n']
    x_number  = np.floor(np.array(data,float)*bins)
    numbers, totals    = np.unique(x_number, return_counts=True)
    density   = totals/(np.sum(totals)*width)
    x_axis    = (numbers + 0.5) * width
    plt.scatter(x_axis,density,s=10,color='black',label='Numerical')
    plt.xlabel('Bins centres')
    plt.ylabel('Density')
    plt.title(f'Invariant Measure for r = {r}')
    plt.xlim(0,1)
    plt.ylim(0,2)
    plt.axhline(1,linestyle='-',linewidth=1,color='red',label='Analytical')
    plt.legend()
    plt.show()
    
'''
Calculate the shannon entropy
'''

def tent_shannon_entropy(r,x,iterations,transients,bins,base='e'):
    data            = Tent_values(r,x,iterations,transients)['X_n']
    values          = np.array(data*bins)
    floor           = np.floor(values)
    numbers, counts = np.unique(floor, return_counts=True)
    prob            = counts/sum(counts)
    clean_prob      = prob[prob>0]
    if base == 'e':
        entropy_i   = clean_prob*np.log(clean_prob)
        entropy     = -sum(entropy_i)
    elif base == 2:
        entropy_i   = clean_prob*np.log2(clean_prob)
        entropy     = -sum(entropy_i)
    return entropy

def tent_shannon_entropy_range(r1,r2,x,iterations,transients,bins,base='e'):
    r_values = np.linspace(r1,r2,1000)
    entropy  = []
    for r in r_values:
        entropy.append(tent_shannon_entropy(r,x,iterations,transients,bins,base))
    plt.scatter(r_values,entropy,marker='.',s=2,color='black')
    plt.grid(True, which='both',linestyle='--',color='grey',linewidth=0.5)
    plt.xlim(r1,r2)
    plt.xlabel('Parameter (r)')
    plt.ylabel('Shannon Entropy')
    plt.title(f'Shannon Entropy for Tent Map with Bins = {bins}')
    plt.show()
        
        

            
