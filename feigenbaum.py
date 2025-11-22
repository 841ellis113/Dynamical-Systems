import numpy
import matplotlib.pyplot as plt

# A program that graphs the logistic map for values of the parameter r between 0 and 4. Above 4
#the logistic map is ill defined. It overlays the lyapunov exponent calculated for each value
#of r. A positive value of the lyapunov exponent at r = r0 indicates that parameter is well
#behaved or non-chaotic. For negative exponents, this indicates that the particulr r value
#is chaotic in nature. The gradient of the Lyapunov exponent graph indicates the sensitivity
#the system is to changes in the parameter. A large absolute value of the exponent means that
#the system predictability is more sensitve to small changes in initial conditions. A smaller
#value indicates that the predictability of the system with tha value of r is less sensitive
#to small changes in initial conditions

#The logistic map
def logistical(r,x):
    x = r*x*(1-x)
    return x

#its derivative
def logistical_derivative(r,x):
    return r*(1-2*x)

#calculating the Lyapunov exponent for a particular initial value of  x0 and parameter r
#the first 5000 itereations are discarded to remove transient results and allow the
#value of x to settle to a more truthful value. The exponent is then calculated using
# the definition of (1/n)(sum(ln(|f'|)). 
def lyapunov(r,x,iterations):
    derivative = []
    for i in range(iterations):
        value = numpy.abs(logistical_derivative(r,x))
        x     = logistical(r,x)
        if i > 5000:
            derivative.append(value)
    D_array = numpy.array(derivative)
    log     = numpy.log(D_array) 
    lyapunov = round((sum(log)/len(log)),2)
    return lyapunov

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

#The next 2 user functions then plot the loogistic map and Lyapunov exponents for different
#values of parameter r, ranging from 0 to 4. After 4, the logistic map breaks down...sort of
def range_logistic(x0,iterations):
    for r in numpy.linspace(0,4,801):
        values = logistic_map_values(r,x0,iterations)
        plt.plot(values[0],values[1],"k.", markersize = 0.1)
    plt.ylim(0,1)
    plt.xlim(0,4)
    plt.title('The Logistic Map Transitioning to Chaos')
    plt.show()

def range_lyapunov(x0,iterations):
    r_values = []
    values   = []
    for r in numpy.linspace(0,4,801):
        values.append(lyapunov(r,x0,iterations))
        r_values.append(r)
    plt.plot(r_values,values)
    plt.show()

#finally overlay both graphs onto the same axis. We can see that at arounr r = 3.57, the value of
    #the Lyapunov exponent becomes +ve, indicating a transition into chaos. This can be seen
    #on the logitic map as n explosion of black data points to the right of this value of r.
def Overlay_plot(x0,iterations):
    lyapunov_values = []
    r_values        = []
    fig,ax1         = plt.subplots()
    for r in numpy.linspace(3.57,4,801):
        values = logistic_map_values(r,x0,iterations)
        lyapunov_values.append(lyapunov(r,x0,iterations))
        r_values.append(values[0])
        ax1.plot(values[0],values[1],"k.", markersize = 0.1, alpha=0.3)
    ax1.set_xlabel('Parameter r')
    ax1.set_ylabel('Logistic map value / x')
    ax1.set_xlim(3.57,4)
    ax1.set_ylim(0,1.1)
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, which='both', linestyle='--', color='lightgray')
    ax2 = ax1.twinx()
    ax2.plot(r_values,lyapunov_values, color='tab:blue', linewidth='0.7')
    ax2.set_ylabel('Lyapunov Exponent', color='tab:blue')
    ax2.tick_params(axis='y',labelcolor='tab:blue')
    #ax2.axhline(0,color = 'red', linestyle='--', linewidth='0.8', label='$\\lambda$ = 0')
    ax2.axvline(3.57, color = 'red', linestyle='--', linewidth='0.8', label = 'r =  3.57')
    plt.suptitle('Overlay of Logistic Map and Lyapunov Exponent at Different Values of r')
    plt.show()

#for a given value of r, iterate enough and discard transients, then determine orbit values
# and determine how many are unique within a tolerance.
def clustering(r,x0,iterations):
    values = logistic_map_values(r,x0,iterations)[1]
    tolerance = 1e-3
    orbits    = []
    while len(values)>0:
        wip = []
        compare = values[0]
        orbits.append(compare)
        for i in values:
            if abs(i-compare)>tolerance:
                wip.append(i)
        values = wip
    return len(orbits)

#determine the number of x or orbit values for the map for a given value of the parameter
#and initial x0 then do this for the range of r between 0<= r <=4

def period_doubling(x0,iterations):
    orbits = []
    r_values = []
    for r in numpy.linspace(0,4,801):
        orbits.append(clustering(r,x0,iterations))
        r_values.append(r)
    unique_x = []
    unique_r = []
    for i in range(len(orbits)):
        if orbits[i] in unique_x:
            continue
        else:
            unique_x.append(orbits[i])
            unique_r.append(r_values[i])
    return unique_x, unique_r

def logistic_period_doubling(x0,iterations):
    xx,rr = period_doubling(x0,iterations)
    for r in numpy.linspace(2.9,4,1001):
        values = logistic_map_values(r,x0,iterations)
        plt.plot(values[0],values[1],"k.", markersize = 0.1)
    plt.ylim(0,1)
    plt.xlim(2.9,4)
    plt.title('The Logistic Map Transitioning to Chaos')
    plt.axvline(rr[1], color='xkcd:periwinkle', linewidth='1')
    plt.text(rr[1],0,'r = 3, Period 2', color = 'xkcd:periwinkle')
    plt.axvline(rr[2], color='xkcd:periwinkle', linewidth='1')
    plt.text(rr[2],0.1,'r = 3.45, Period 4', color = 'xkcd:periwinkle')
    plt.axvline(rr[3], color='xkcd:periwinkle', linewidth='1')
    plt.text(rr[3],0.21,'r = 3.54, Period 8', color = 'xkcd:periwinkle')
    plt.axvline(rr[4], color='xkcd:periwinkle', linewidth='1')
    plt.text(rr[4],0.7,'r = 3.56, Period 16', color = 'xkcd:periwinkle')
    plt.axvline(rr[5], color='xkcd:periwinkle', linewidth='1')
    plt.text(rr[5],.95,'r = 3.57, Feigenbaum Point', color = 'xkcd:periwinkle')
    plt.show()






    
