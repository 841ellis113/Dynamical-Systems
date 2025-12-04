import numpy as numpy
import matplotlib.pyplot as plt
import random 
import seaborn as sns

'''
The aim of this script is to calculate the box dimension of thr logistic map attractor. The
box counting or box dimension of a map is related to how many boxes of side length e
are needed to completly cover all points of map. Once this is done, we repeat the process with
smaller sized boxes. We continue this proces and take the limit of log(N(e))/log(1/e)
as e approaches 0. The box dimension is a measure of how much of the available space is
'used' by the logistic map at a given orbit and is a space filling measurement. Knowing this,
we can say that for orbits that are stable and oscillate between a small number of values,
the orbit uses a very small amount of the domain [0,1] to describe this points, the box
counting dimension should be very close to 0. The more chaotic the system is in, the more
of the domain is used with each successive iteration, and so the box counting dimension
will be closer to 1 (the Logistic Map is a 1D, thus the highest it can be is 1 if all the
domain is needed to describe the orbits).
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
plotting. We calculate the exponent using the definition of (1/n)(sum(ln(|f'|)). The input
values would be parameter value r, initial x value x0, the number of iterations and the
number of iterations to allow before recording data (transients) k or (r,x,i,t)
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
            
'''
Then we graph the ranges of x as we vary the parameter r.
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

'''
To calculate the box counting dimension we first need to determine how many boxes or 'lines'
that are within the domain for example if the box size e is 0.1, then in the domain there
will be 1/0.1 = 10 boxes. If e is 0.03, then there will be 1/0.03 = 300 boxes. Next we need
to determine which box each point on the orbit is covered by. If the value of a point is
0.23 and e = 0.05, then the point fits within the 4th or 5th box depending on how we label
the box numbers. We will start box labels from 0 to fit in with python standards. Thus, this
point will fit into the 4th box. We do the same for all x points in the orbit. We achieve
this by dividing all values within the array by the box size, then 'floor' the values to
determine which box it fits in.
Next we determine how many unique boxes are needed to cover all the orbit points and sum.
Then we calculate the the number of boxes used and output log(N) and log(1/e).

'''

def box_counting(X,epsilon):
    data        = numpy.clip(X,0,1-1e-15)
    box_fitting = numpy.floor(data/epsilon)
    N_total     = numpy.unique(box_fitting)
    N           = len(N_total)
    return numpy.log(N), numpy.log(1/epsilon)

'''
Expand to determine a range of logN and loge values for a particular set of (r,x,i,t). To
get the different ranges of epsilon. In order to get the value of D we must extract the
slope of a log-log plot, but we msu do in a range that will provide an accurate answeer.
To do this we must negate large epsilon values at the start of the graph and ignore
the values at very small epsilon. This is because as the 'boxes' become smaller, each point
on the orbit will fit into their individual box. No matter how small we make epsilon, the number
of points per box will be 1 and the log-log curve will begin to flatten. So as the number of
boxes approaches the number of points, we dont use this regime.
We want the total number of boxes to be less than the total number of data points and the
number of boxes is the domain(1) devided by the box length. Combining, this produces the
condition points*epsilon > 1 or > 1.1 to stay away from the boundary. 
'''

def box_counting_values(r,x0,iterations,transients):
    data      = logistic_map_values(r,x0,iterations,transients)[1]
    eps       = 10/(iterations - transients)
    epsilon   = numpy.linspace(0.01,eps,3001)
    logN_vals = []
    loge_vals = []
    for e in epsilon:
        logN, loge = box_counting(data,e)
        logN_vals.append(logN)
        loge_vals.append(loge)
    D = round(numpy.polyfit(loge_vals,logN_vals,1)[0],2)
    sns.regplot(x=loge_vals,y=logN_vals,ci=None, line_kws={'color':'red',\
    'linestyle':'--','linewidth':0.75,'label':f'D={D}'},scatter_kws={'color':'black','s':0.7,\
    'alpha':0.2,'marker':'.'})
    plt.title(f'Plot of Number of Boxes (N) and width size $\epsilon$, r={r}')
    plt.xlabel('Log(1/$\epsilon$)')
    plt.ylabel('Log N($\epsilon$)')
    plt.grid(True, which='both',linestyle='--',linewidth=0.1,color='lightgrey')
    plt.legend()
    plt.show()
    return D
    

'''
takes x0, iterations and transient levels to output a range of values for the box counting
dimension through out the range of r. manually adds tghe point r=3.56995 to calculate
dimension at the point where chaos begins or Feigenbaum point
'''

def range_box_counting_values(r1,r2,x0,iterations,transients):
    eps       = 10/(iterations - transients)
    epsilon   = numpy.linspace(0.01,eps,3001)
    feigenbaum= numpy.array([3.56995])
    r_param   = numpy.linspace(r1,r2,107)
    parameter = numpy.sort(numpy.concatenate([r_param,feigenbaum]))
    r_values  = []
    dimension = []
    for r in parameter:
        data      = logistic_map_values(r,x0,iterations,transients)[1]
        r_values.append(r)
        logN_vals = []
        loge_vals = []
        for e in epsilon:
            logN, loge = box_counting(data,e)
            logN_vals.append(logN)
            loge_vals.append(loge)
        D     = round(numpy.polyfit(loge_vals,logN_vals,1)[0],2)
        dimension.append(D)
    plt.scatter(r_values,dimension,s=50,color='black',marker='o')
    plt.xlim(r1,r2)
    plt.ylim(0,1)
    plt.xlabel('Parameter (r)')
    plt.ylabel('Box Counting Dimension (D)')
    plt.title('Range of Box Counting Dimension')
    plt.show()
    return r_values, dimension

def Overlay_plot(r1,r2,x0,iterations,transients):
    r_values  = numpy.linspace(r1,r2,1001)
    eps       = 10/(iterations - transients)
    epsilon   = numpy.linspace(0.01,eps,3001)
    feigenbaum= numpy.array([3.56995])
    r_param   = numpy.linspace(r1,r2,72)
    parameter = numpy.sort(numpy.concatenate([r_param,feigenbaum]))
    r_points  = []
    dimension = []
    fig,ax1   = plt.subplots()
    for r in r_values:
        values = logistic_map_values(r,x0,iterations,transients)
        ax1.plot(values[0],values[1],"k.", markersize = 0.1, alpha=0.3)
    for r in parameter:
        data      = logistic_map_values(r,x0,iterations,transients)[1]
        r_points.append(r)
        logN_vals = []
        loge_vals = []
        for e in epsilon:
            logN, loge = box_counting(data,e)
            logN_vals.append(logN)
            loge_vals.append(loge)
        D     = round(numpy.polyfit(loge_vals,logN_vals,1)[0],2)
        dimension.append(D)
    ax1.set_xlabel('Parameter r')
    ax1.set_ylabel('Orbit Value (x)')
    ax1.set_xlim(r1,r2)
    ax1.set_ylim(0,1)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', linestyle='--', color='lightgray')
    ax2 = ax1.twinx()
    ax2.scatter(r_points,dimension, color='red', s=30, marker='x')
    ax2.set_ylabel('Box Counting Dimension', color='red')
    ax2.tick_params(axis='y',labelcolor='red')
    ax2.set_ylim(0,1)
    plt.suptitle('Overlay of Logistic Map and Box Dimension at Different Values of r')
    plt.show()
