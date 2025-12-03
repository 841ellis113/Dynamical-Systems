import numpy
import matplotlib.pyplot as plt
import random

'''
A program that graphs the logistic map for values of the parameter r between 0 and 4. Above 4
the logistic map is ill defined. It overlays the lyapunov exponent calculated for each value
of r. A positive value of the lyapunov exponent at r = r0 indicates that parameter is well
behaved or non-chaotic. For negative exponents, this indicates that the particulr r value
is chaotic in nature. The gradient of the Lyapunov exponent graph indicates the sensitivity
the system is to changes in the parameter. A large absolute value of the exponent means that
the system predictability is more sensitve to small changes in initial conditions. A smaller
value indicates that the predictability of the system with tha value of r is less sensitive
to small changes in initial conditions
'''

#The logistic map
def logistic(r,x):
    return r*x*(1-x)

#its derivative
def logistic_derivative(r,x):
    return r*(1-2*x)

'''
calculate the values of the logistical map with an initial x0 and parameter r, only capturing dat
after removing the transients, we also add a small value to
the value of x0 as to remove synchronis artifacts as well as taking a random sample before
plotting. We calculate the exponent using the definition of (1/n)(sum(ln(|f'|)).
'''

def logistic_map_values(r,x0,iterations,transients):
    x = x0 + random.uniform(0,1)*1e-5
    rawx_values   = []
    rawr_values   = []
    sample        = int((iterations-transients)/2)
    for i in range(iterations):
        x = logistic(r,x)
        if i > transients:
            rawx_values.append(x)
            rawr_values.append(r)
    x_values      = random.sample(rawx_values,sample)
    r_values      = random.sample(rawr_values,sample)
    return r_values, x_values

def Lyapunov_value(r,x0,iterations,transients):
    x          = x0 + random.gauss()*1e-5
    d_values   = []
    for i in range(iterations):
        d = numpy.abs(logistic_derivative(r,x))
        x = logistic(r,x)
        if i > transients:
            d_values.append(d)
    d_array    = numpy.array(d_values)
    log_d      = numpy.log(d_array)
    Lyapunov   = sum(log_d)/len(log_d)
    return Lyapunov
            
'''
Then we graph the ranges of x as we vary the parameter r and we also grap the Lyapunov
values over the range of r between 0 and 4.
'''

def graph_logistic(r1,r2,x0,iterations,transients):
    for r in numpy.linspace(r1,r2,1001):
        values = logistic_map_values(r,x0,iterations,transients)
        plt.plot(values[0],values[1],"k.", markersize = 0.1,alpha = 0.15)
    plt.ylim(-1,1)
    plt.xlim(r1,r2)
    plt.title('The Logistic Map Transitioning to Chaos')
    plt.xlabel('Parameter (r)')
    plt.ylabel('Orbit Values (x)')
    plt.grid(True,which='both',linewidth=0.5,color='grey',linestyle='--')
    plt.show()

def range_lyapunov(r1,r2,x0,iterations,transients):
    r_values = []
    values   = []
    for r in numpy.linspace(r1,r2,1001):
        values.append(Lyapunov_value(r,x0,iterations,transients))
        r_values.append(r)
    plt.plot(r_values,values)
    plt.xlabel('Parameter (r)')
    plt.ylabel('Lyapunov Exponent ($\\lambda$)')
    plt.title('Lyapunov Values')
    plt.grid(True,which='both')
    plt.xlim(r1,r2)
    plt.ylim(-5,1)
    plt.axhline(0,color = 'black', linestyle='--',linewidth=0.5)
    plt.axhspan(0,1,r1,r2,color='black',alpha=0.1,label='$\\lambda$>0')
    plt.axhspan(-5,0,r1,r2,color='red',alpha=0.1,label='$\\lambda$<0')
    plt.legend()
    plt.show()

'''
Finally we overlay both the lyapunov and the bifurcation maps on top of each other.
'''

def Overlay_plot(r1,r2,x0,iterations,transients):
    lyapunov= []
    r_values        = []
    fig,ax1         = plt.subplots()
    for r in numpy.linspace(r1,r2,1001):
        values = logistic_map_values(r,x0,iterations,transients)
        lyapunov.append(Lyapunov_value(r,x0,iterations,transients))
        r_values.append(values[0])
        ax1.plot(values[0],values[1],"k.", markersize = 0.1, alpha=0.3)
    ax1.set_xlabel('Parameter r')
    ax1.set_ylabel('Orbit Value (x)')
    ax1.set_xlim(r1,r2)
    ax1.set_ylim(0,1)
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, which='both', linestyle='--', color='lightgray')
    ax2 = ax1.twinx()
    ax2.plot(r_values,lyapunov, color='tab:blue', linewidth='0.7')
    ax2.set_ylabel('Lyapunov Exponent', color='tab:blue')
    ax2.tick_params(axis='y',labelcolor='tab:blue')
    #ax2.axvline(3.57, color = 'red', linestyle='--', linewidth='0.8', label = 'r =  3.57')
    ax2.set_ylim(-5,1)
    ax2.axhspan(0,1,color='black',alpha=0.1,label='$\\lambda$>0')
    ax2.axhspan(-5,0,color='red',alpha=0.1,label='$\\lambda$<0')
    plt.suptitle('Overlay of Logistic Map and Lyapunov Exponent at Different Values of r')
    plt.legend(loc='lower left')
    plt.show()
