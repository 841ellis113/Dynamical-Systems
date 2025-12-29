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

'''
Calculate the Shannon Entropy of the logistic map for a given value of the parameter r
'''

def logistic_shannon_entropy(r,x,iterations,transients,bins,base='e'):
    rdata, xdata = logistic_map_values(r,x,iterations,transients)
    data   = numpy.array(xdata,float)
    allocation = numpy.floor(data*bins)
    unique, counts = numpy.unique(allocation, return_counts=True)
    prob = counts/sum(counts)
    prob = prob[prob>0]
    if base == 'e':
        entropy_i = prob*numpy.log(prob)
        entropy   = -1*sum(entropy_i)
    elif base == 2:
        entropy_i  = prob*numpy.log2(prob)
        entropy   = -1*sum(entropy_i)
    return entropy

def shannon_entropy_range(r1,r2,x,iterations,transients,bins,base='e'):
    r_range = numpy.linspace(r1,r2,2500)
    r_values = []
    entropy = []
    for r in r_range:
        r_values.append(r)
        ent  = logistic_shannon_entropy(r,x,iterations,transients,bins,base)
        entropy.append(ent)
    plt.scatter(r_values,entropy, marker='.', s=3,color='red')
    plt.grid(True, which = 'both', linestyle ='--', linewidth=1, color='grey')
    plt.title(f'Variation of Shannon Entropy for Logistic Map with r for {bins} Bins')
    plt.xlim(0,4)
    plt.xlabel('Parameter (r)')
    plt.ylabel('Shannon Entropy')
    plt.show()

        
