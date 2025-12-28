import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
The sine map is another 1d map that exhibits chaotic behaviour. the discrete map is iterated
through the function rsin(pi*x) for xE [0,1] and rE [0,1]. We also define the derivative for
the Lyapunov exponents
'''

def sin_map(r,x):
    return r*(np.sin(np.pi*x))

def sin_derivative(r,x):
    return r*np.pi*(np.cos(np.pi*x))

'''
Next is to then to create a dataframe that stores a number of iterations, discarding
transients  and outputs the orbit values as well as the lyapunov exponent at each x value
and also the time-delay embedded matrix for t time delays columns. The time delay value
is set to 1 by default, meaning if no chaning in argument tau, the function will
automatically output X(n+1) values only. This will be the backbone of further functions
for graphing
'''

def Sin_embedded_matrix(r,x,iterations,transients,tau=1,remove=True):
    raw_steps    = []
    raw_x_values = []
    Insta_Lya    = []
    for i in range(iterations+1):
        raw_steps.append(i)
        raw_x_values.append(x)
        wip = np.log(np.abs(sin_derivative(r,x)))
        Insta_Lya.append(wip)
        x = sin_map(r,x)
    if remove == True:
        steps    = raw_steps[transients:]
        x_values = raw_x_values[transients:]
        Lyapunov = Insta_Lya[transients:]
    else:
        steps    = raw_steps
        x_values = raw_x_values
        Lyapunov = Insta_Lya
    data = pd.DataFrame({'Iterations':steps,'Lyapunov':Lyapunov,'r':r,'X_n':x_values})
    for t in range(1,1+tau):
        column = []
        for L in range(len(data)):
            try:
                column.append(data['X_n'].iloc[L+t])
            except IndexError:
                column.append(None)
        data[f'X_n+{t}'] = column
    data = data.dropna()
    return data

'''
Creating a time series graph
'''

def time_series(r,x,iterations,transients):
    data = Sin_embedded_matrix(r,x,iterations,transients,tau=1,remove=True)
    plt.plot(data['Iterations'],data['X_n'],color='black',marker='x',linewidth=0.5,linestyle='--')
    plt.title("Time Series for Sine Map")
    plt.xlabel('Time Step')
    plt.ylabel('Orbit Values')
    plt.show()

'''
Create a figure that has 5 time series plots (2x3) each with a different values of r. The
5 values of r used must be in the form of a list and inputed as r_list.
'''

def time_series_multiple(r_list,x,iterations,transients):
    fig,ax = plt.subplots(5)
    for i, r in enumerate(r_list):
        data = Sin_embedded_matrix(r,x,iterations,transients,tau=1,remove=True)
        ax[i].plot(data['Iterations'],data['X_n'],color='black',marker='x',linewidth=0.5,
                   linestyle='--',label=f'r={r}')
        ax[i].set_ylim(0,1)
        ax[i].set_xlim(transients,iterations-1)
        ax[i].legend()
        ax[i].grid(True, which='both',linestyle='--',linewidth=0.7,color='grey')
        if i != 4:
            ax[i].tick_params(axis='x', labelbottom=False)
    plt.suptitle('Time Series for Sine Map')
    plt.xlim(transients,iterations-1)
    fig.supxlabel('Time Step')
    fig.supylabel('$X_n$')
    plt.show()

'''
Create a bifurcation diagram for the Sine map
'''

def sine_bifurcation(r1,r2,x,iterations,transients):
    r_values = np.linspace(r1,r2,1200)
    for r in r_values:       
        data = Sin_embedded_matrix(r,x,iterations,transients)
        plt.scatter(data['r'],data['X_n'],s=0.6,color='black',alpha=0.05)
    plt.title('Sine Map Bifurcation')
    plt.xlabel('Parameter (r)')
    plt.xlim(0,1)
    plt.ylabel('Orbit Values (x)')
    plt.ylim(0,1)
    plt.show()
    
'''
Claculating Lyapunov exponent over range r1 to r2
'''

def Lyapunov(r1,r2,space,x,iterations,transients):
    lambdaa = []
    r_parameter = []
    r_values = np.linspace(r1,r2,space)
    for r in r_values:
        r_parameter.append(r)
        data = Sin_embedded_matrix(r,x,iterations,transients)
        wip = data.groupby('r')['Lyapunov'].mean()
        lambdaa.append(float(wip.iloc[0])) 
    plt.scatter(r_parameter,lambdaa,s=1.1,color='red')
    plt.grid(True,which='both',color='grey',linewidth=1,linestyle='--')
    plt.axhline(0, color='black', linewidth = 0.7)
    plt.xlabel('Parameter (r)')
    plt.xlim(r1,r2)
    plt.ylabel('$\lambda$')
    plt.ylim(-4,1)
    plt.title('Variation of Lyapunov Exponent with r')
    plt.show()

'''
Creating return maps
'''

def return_maps(r,x,iterations,transients,tau=1,remove=True):
    data = Sin_embedded_matrix(r,x,iterations,transients,tau=tau)
    if tau == 1:
        plt.scatter(data['X_n'],data['X_n+1'],s=1,color='black',alpha=0.5)
        plt.show()
    else:
        fig,ax = plt.subplots(1,tau)
        for t in range(tau):
            ax[t].scatter(data['X_n'],data[f'X_n+{t+1}'],s=1,color='black',alpha=0.5)
            ax[t].set_title(f't = {t+1}')
        fig.suptitle('Return Maps for the Sine Map')
        fig.supylabel('$X_{n+t}$')
        fig.supxlabel('$X_n$')
        plt.show()

'''
Calculating the Invariant density of the sine map
'''

def sine_invariant(x):
    Pi = np.pi
    return 1/(Pi*np.sqrt(x*(1-x)))

def sine_invariant_density(r,x,iterations,transients,bins):
    width = 1/bins
    data      = Sin_embedded_matrix(r,x,iterations,transients)['X_n']
    x_number  = np.floor(np.array(data,float)*bins)
    numbers, totals    = np.unique(x_number, return_counts=True)
    density   = totals/(np.sum(totals)*width)
    x_axis    = (numbers + 0.5) * width
    x_range   = np.linspace(0,1,bins)
    y_range   = [sine_invariant(x) for x in x_range]
    plt.scatter(x_axis,density,s=1,color='black',label='Numerical',alpha=0.6
                )
    plt.xlabel('Bins centres')
    plt.ylabel('Density')
    plt.title(f'Invariant Measure for r = {r}')
    plt.xlim(0,1)
    plt.ylim(0,15)
    plt.plot(x_range,y_range,linewidth=1,color='red',label='Analytical')
    plt.legend()
    plt.show()
