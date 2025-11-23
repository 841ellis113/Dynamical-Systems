import numpy
import matplotlib.pyplot as plt

#The logistic map
def logistical(r,x):
    x = r*x*(1-x)
    return x

#calculate the values of the logistical map with an initial x0 and parameter r, only capturing dat
#after the first 5000 iterations to remove transient results
def logistic_map_values(r,x,iterations):
    x_values   = []
    r_values   = []
    for i in range(iterations):
        x = logistical(r,x)
        if i > 5000:
            x_values.append(x)
            r_values.append(r)
    return r_values, x_values

#function then plot the logistic map for different
#values of parameter r, ranging from 0 to 4. After 4, the logistic map breaks down...sort of
def range_logistic(x0,iterations):
    for r in numpy.linspace(0,4,1001):
        values = logistic_map_values(r,x0,iterations)
        plt.plot(values[0],values[1],"k.", markersize = 0.1)
    plt.ylim(0,1)
    plt.xlim(0,4)
    plt.title('The Logistic Map Transitioning to Chaos')
    plt.show()

#next we split the domain of [0,1], then break down the doamin into regions of size epsilon
#We divide the point by epsilon to get the index in which box that point is located. We also
#clip the data points t ensure all points will be 'boxed'. Then the total number of boxes
# needed to cover all the data points is used to determine the box dimension.
#Inputs are array of data points and epsilon value.
def box_counting(data,epsilon):
    points = numpy.clip(data,0,1-1e-15)
    box_numbers = numpy.floor(points / epsilon)
    numbers     = numpy.unique(box_numbers)
    Nlog        = numpy.log(len(numbers))
    epslog      = (numpy.log(1/epsilon))
    return epslog, Nlog

#this function takes the argument of a specific value of r and initial x. It then calculates
#Log(N) and Log(1/epsilon) and returns them for smaller and smaller values of epsilon
#as well as the values of epsilon used. 
def range_box_counting(r,x,iterations):
    eps       = 0.1
    logN = []
    logEps   = []
    EPS    = []
    data = logistic_map_values(r,x,iterations)[1]
    while eps >1e-3:
        EPS.append(eps)
        e,N = box_counting(data, eps)
        logN.append(N)
        logEps.append(e)
        eps /= 2
    return logN, logEps, EPS

#this then plots the Log(N) on the y axis and Log(1/epsilon) on the x axis. It also plots
#the Box Counting Dimension against Log(1/epsilon)for a given value of r
def dimension_graph(r,x,iterations):
    logN,logEps,EPS = range_box_counting(r,x,iterations)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(logEps,logN)
    ax[0].set_title(f'Graph of Log(N) vs Log(1/$\epsilon$) (r={r})')
    ax[0].set_xlabel('Log(1/$\epsilon$)')
    ax[0].set_ylabel('Log(N)')
    ax[0].tick_params(axis='y', labelcolor='k')
    LOGN = numpy.array(logN)
    LOGEPS = numpy.array(logEps)
    Dimension = LOGN/LOGEPS
    ax[1].plot(LOGEPS,Dimension)
    ax[1].set_xlabel('Log(1/$\epsilon$)')
    ax[1].tick_params(axis='y',labelcolor='tab:blue')
    ax[1].set_ylabel('Box Dimension, D')
    ax[1].set_title(f'Box Counting Dimension, D, as $\epsilon$ Decreases (r={r})')
    plt.show()

#calculates the average value of the box counting dimension over the values of LogN and
#Log(1/epsilon) from range_box_counting for a given r.
def dimension(r,x,iterations):
    logN,logEps,EPS = range_box_counting(r,x,iterations)
    length = len(logN)
    D = []
    for i in range(length-1):
        D.append((logN[i+1]-logN[i])/(logEps[i+1]-logEps[i]))
    dimension = sum(D)/len(D)
    return dimension

#plots the box_counting dimension using for a given range of the paramenter r. the inputs are
#r1 and r2, the range of r, x0 for the initial value of x and the number of iterations
def plot_dimension(r1,r2,x0,iterations):
    y = []
    x = []
    for r in numpy.linspace(r1,r2,1000):
        y.append(dimension(r,x0,iterations))
        x.append(r)
    plt.plot(x,y)
    plt.xlabel('Parameter r')
    plt.xlim(3.2,4)
    plt.ylim(0,1)
    plt.ylabel('Box Dimension, D')
    plt.title('Variation of Box Dimension with Parameter r')
    plt.show()

#finally we overlay the bifurcation plot with the box dimension
def overlay_box(r1,r2,x0,iterations):
    dimensions      = []
    r_values        = []
    fig,ax1         = plt.subplots()
    for r in numpy.linspace(r1,r2,801):
        values = logistic_map_values(r,x0,iterations)
        dimensions.append(dimension(r,x0,iterations))
        r_values.append(values[0])
        ax1.plot(values[0],values[1],"k.", markersize = 0.1, alpha=0.3)
    ax1.set_xlabel('Parameter r')
    ax1.set_xlim(r1,r2)
    ax1.set_ylabel('Logistic map value / x')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, which='both', linestyle='--', color='lightgray')
    ax2 = ax1.twinx()
    ax2.plot(r_values,dimensions, color='tab:blue', linewidth='0.7')
    ax2.set_ylabel('Box Dimension', color='tab:blue')
    ax2.tick_params(axis='y',labelcolor='tab:blue')
    plt.suptitle('Overlay of Logistic Map and Box Dimension at Different Values of r')
    plt.show()
