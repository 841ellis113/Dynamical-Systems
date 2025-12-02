import numpy
import matplotlib.pyplot as plt
import random
'''
This is a script that maps the logistic equation for the ranges of r between 0 and 4
'''

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
    x = x0 + random.uniform(0,1)*1e-8
    rawx_values   = []
    rawr_values   = []
    sample        = int((iterations-transients)/2)
    for i in range(iterations):
        x = logistic(r,x)
        if i > transients:
            rawx_values.append(x)
            rawr_values.append(r)
    x_values = random.sample(rawx_values,sample)
    r_values = random.sample(rawr_values,sample)
    return r_values, x_values

'''
Then we graph the ranges of x as we vary the parameter r.
'''

def graph_logistic(r1,r2,x0,iterations,transients):
    for r in numpy.linspace(r1,r2,1001):
        values = logistic_map_values(r,x0,iterations,transients)
        plt.plot(values[0],values[1],"k.", markersize = 0.1,alpha = 0.15)
    plt.ylim(0,1)
    plt.xlim(r1,r2)
    plt.title('The Logistic Map Transitioning to Chaos')
    plt.xlabel('Parameter (r)')
    plt.ylabel('Orbit Values (x)')
    plt.grid(True,which='both',linewidth=0.5,color='grey',linestyle='--')
    plt.show()
