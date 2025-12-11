import numpy 
import matplotlib.pyplot as plt
import random 
import seaborn as sns


#The logistic map
def logistic(r,x):
    return r*x*(1-x)
'''
calculate the values of the logistical map with an initial x0 and parameter r, only capturing dat
after removing the transients, we also add a small value to
the value of x0 as to remove synchronis artifacts as well as taking a random sample before
plotting. 
'''
def logistic_map_values(r,x0,iterations,transients):
    x = x0 + random.uniform(0,1)*1e-5
    x_values   = []
    for i in range(iterations):
        x = logistic(r,x)
        if i > transients:
            x_values.append(x)
    return x_values
            
'''
We can produce a matrix formed from vectors that consist of iteration steps from within the time
series. The number of different column vectors is the embedding dimension and the number
of iterations taken before capturing data points into the column vector is known as the
time delay, i.e. if m = 2 and t=3 means we construct a matrix with 2 vectors and each
point within the vectors is 3 steps from the previous
'''
def embedded_matrix(m,t,r,x0,iterations,transients):
    x_data = logistic_map_values(r,x0,iterations,transients)
    stopper = len(x_data) - m*t
    matrix = numpy.zeros([stopper,m],float)
    for i in range(m):
        column = numpy.array(x_data[i*t:stopper+i*t])
        matrix[:,i] = column
    return matrix
'''
to begin with, we plot the simple time series against iteration step. The time series
graph simply returns the 'trajectory' taken by the map and after iterating for N step
returns the final 50 or so steps as not to overcrowed the graph.
'''
def time_series_map(r,x0,iterations,transients):
    data    = logistic_map_values(r,x0,iterations,transients)
    rvalues = transients + numpy.array(list(range(len(data))))
    #plt.scatter(rvalues,data,color='red',marker='x',s=2,label=f'r = {r}')
    plt.plot(rvalues,data,color='red',marker='x',markersize=3,linewidth=0.4,linestyle='--')
    plt.grid(True, which='both',color='black',linestyle='--',linewidth=0.2)
    plt.xlabel('Step Number (n)')
    plt.ylim(0,1)
    plt.ylabel('X')
    plt.title(f'Times series of the Logistic Map (r={r})')
    plt.show()
'''
Plot a time series for a range of r values and plot them on a 2x3 plot.
'''
def compare_TS_2x3_plot(r1,r2,r3,r4,r5,r6,x0,iterations,transients):
    fig, axes  = plt.subplots(2,3)
    parameter = [[r1,r2,r3],[r4,r5,r6]]
    for m in range(2):
        for n in range(3):
            data      = logistic_map_values(parameter[m][n],x0,iterations,transients)
            rvalues   = transients + numpy.array(list(range(len(data))))
            axes[m][n].plot(rvalues,data,marker='x',linestyle='--',markersize=5,linewidth=0.5)
            axes[m,n].grid(True,which='both',color='black',linestyle='--',linewidth=0.2)
            axes[m,n].set_ylim(0,1)
            axes[m,n].set_title(f'r = {parameter[m][n]}')
    fig.supxlabel('Iteration Step')
    fig.supylabel('X value')
    plt.suptitle('Time Series of the Logistic Map at Different r Values')
    plt.show()

'''
Next we produce return maps or poincare plots. These graph points X against the next point
on the trajectory. We also produce a function to do this over multiple values of r
'''

def two_D_Poincare_plot(t,r,x0,iterations,transients):
    matrix = numpy.round(embedded_matrix(2,t,r,x0,iterations,transients),3)
    plt.scatter(matrix[:,0],matrix[:,1],s=5,marker='x',color='black')
    plt.xlabel('$X_n$')
    plt.ylabel(rf'$X_{{n+{t}}}$')
    plt.title('Return Map of the Logistic Map')
    plt.show()
    #return matrix

def compare_poincare_plot(m,t,r1,r2,r3,r4,r5,r6,x0,iterations,transients):
    fig, axes  = plt.subplots(2,3)
    parameter = [[r1,r2,r3],[r4,r5,r6]]
    for i in range(2):
        for n in range(3):
            matrix = numpy.round(embedded_matrix(m,t,parameter[i][n],x0,iterations,transients),3)
            axes[i,n].scatter(matrix[:,0],matrix[:,1],s=5,marker='x',color='black')
            axes[i,n].grid(True,which='both',color='black',linestyle='--',linewidth=0.2)
            axes[i,n].set_ylim(0,1)
            axes[i,n].set_title(f'r = {parameter[i][n]}')
    fig.supxlabel('$X_n$')
    fig.supylabel(rf'$X_{{n+{t}}}$')
    plt.suptitle('Poincare plots of the Logistic Map at Different r Values')
    plt.show()

'''
a function that takes an embedded matrix and plots the return map for x, x+1t, x+2t... so
that it can be seen how many steps it takes for the breakdon of accurate predicatability.
'''

def multiple_time_delay(m,t,r,x0,iterations,transients):
    matrix = embedded_matrix(m,t,r,x0,iterations,transients)
    fig,ax = plt.subplots(1,m-1)
    for i in range(m-1):
        ax[i].scatter(matrix[:,0],matrix[:,i+1],s=2,color='black')
        ax[i].set_title(rf'$X_{{n+{i+1}}}$')
    fig.suptitle('Return Map (time delayed)')
    fig.supylabel('$X_{{{n+i}}}$')
    fig.supxlabel('$X_n$')
    plt.show()









            
