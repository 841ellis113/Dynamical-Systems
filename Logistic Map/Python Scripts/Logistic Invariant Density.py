import numpy 
import matplotlib.pyplot as plt
import random 
import seaborn as sns

'''
define the logistic map and return an array of x values
'''

def logistic(r,x):
    return r*x*(1-x)

def logistic_map_values(r,x0,iterations,transients):
    x = x0 + random.uniform(0,1)*1e-5
    x_values   = []
    for i in range(iterations):
        x = logistic(r,x)
        if i > transients:
            x_values.append(x)
    return x_values


'''
calculate the invariant density of the logistic map for parameter r. Do this by allocating
each point into a designated bin. To do this, the x points are divided by the quantity
(1/bins) where bins is the number of bins. this number is rounded down or floored so that
the x point is now a value between 0 and bins and this represents the which bin the x point
belongs to. To do this, the floor function is used.
'''

def Invariant_Density(r,x0,iterations,transients,bins,graph=True):
    width     = 1/bins
    data      = logistic_map_values(r,x0,iterations,transients)
    x_number  = numpy.floor(numpy.array(data,float)*bins)
    numbers, totals    = numpy.unique(x_number, return_counts=True)
    density   = totals/(numpy.sum(totals)*width)
    x_axis    = (numbers + 0.5) * width
    plt.scatter(x_axis,density,s=10,color='black',label='Numerical')
    plt.xlabel('Bins centres')
    plt.ylabel('Density')
    plt.title(f'Invariant Measure for r = {r}')
    plt.xlim(0,1)
    plt.ylim(0,20)
    if graph == True:
        plt.show()
    return density

'''
next we create a program that will plot 6 different plots with increasing bin numbers to
show the increase in resolution of the results for the parameter value of r = 4 and add the
analytic version of the invariant density for comparison. This is of the form
p = 1/(pi*sqrt(x(1-x)))
'''

def analytical(x):
    return 1/(numpy.pi*(numpy.sqrt(x*(1-x))))

def analytical_range(points,graph=True):
    x_range = numpy.linspace(0,1,points)
    y_range = analytical(x_range)
    plt.scatter(x_range,y_range,s=2,color='red',alpha=0.3,label='Analytical')
    if graph == True:
        plt.show()

def Analytical_numerical_overlay(r,x0,iterations,transients,bins,points,graph=False):
    density = Invariant_Density(r,x0,iterations,transients,bins,graph)
    analytical_range(points,graph)
    plt.grid(True, which='both',color='grey',linewidth=0.5,linestyle='--')
    plt.legend()
    plt.show()
    
'''
Note that Invariant_Density and analytical_range have a 'switch' (graph==True), that prints
graph if these are used as stand-alone-functions. If used as part of the Analytical_numberical
function, this is switched to False or off so that the graphs do output until further
down the code when needed.
'''

