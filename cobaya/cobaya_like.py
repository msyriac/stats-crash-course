import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

np.random.seed(100)

# # Generic MCMC with cobaya

# Say that we have a set of observations, a parameterized theory model for the observations, a likelihood function and a set of priors. How can we quickly set up a Python workflow for inferring parameters in such a way that the process can be hybrid OpenMP-MPI parallelized with as little pain as possible? This is what cosmologists typically need (since e.g. Boltzmann codes are OpenMP enabled). For reasonably simple posteriors (no bimodalities, less than 20 parameters, etc.) I think this is easiest with Metropolis-Hastings with cobaya. Here, I show how to dive right into it fast.  Note that this example doesn't utilize any of the special features Cobaya has for parameters of Boltzmann codes like CAMB, but is instead used here for as agnostic a likelihood as possible.
# 
# For demo purposes, let's first set up a simple model to simulate. I'll use the same example from the bayesian inference section of my Stats Crash Course. The main difference from there is that we will be using cobaya instead of emcee (and getdist for plots), but I also replace the bimodal noise distribution with a simple Gaussian one. This latter difference doesn't really matter since we have always worked with binned data, which get Gaussianized (central limit theorem) regardless of how complicated the original noise distribution was.

# ## Generate data

from scipy.stats import binned_statistic

num_bins = 20

def model(x,a,b,c):
    return np.sin(a*x) + b*x +c


true_a = 10
true_b = 2
true_c = -3

def generate_data():
    Npoints = 1000
    x = np.linspace(-1,1,Npoints)
    return model(x,true_a,true_b,true_c) + np.random.normal(scale=10,size=Npoints)
    
xs = np.linspace(-1,1,1000)
bin_edges = np.linspace(-1,1,num_bins)
bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.

true_model_fine = model(xs,true_a,true_b,true_c)
binned_true = binned_statistic(xs,true_model_fine,bins=bin_edges,statistic=np.mean)[0]

data = generate_data()
binned_data = binned_statistic(xs,data,bins=bin_edges,statistic=np.mean)[0]
binned_var = binned_statistic(xs,data,bins=bin_edges,statistic=np.var)[0]
binned_count = binned_statistic(xs,data,bins=bin_edges,statistic=np.size)[0]
binned_sigma = np.sqrt(binned_var/binned_count)

xs = np.linspace(-1,1,1000)
# plt.plot(xs,data,alpha=0.2,color="C0")
# plt.plot(xs,true_model_fine,color="C1",lw=3)
# plt.scatter(bin_centers,binned_true,marker="x",s=128,color='k')
# plt.errorbar(bin_centers,binned_data,yerr=binned_sigma,marker="o",color="C2",ls="none")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# ## Define a likelihood

y = binned_data
yerr = binned_sigma
def lnlike(a,b,c):
    theory = binned_statistic(xs,model(xs,a,b,c),bins=bin_edges,statistic=np.mean)[0]
    residual = y-theory
    inv_sigma2 = 1.0/(yerr**2.)
    return -0.5*(np.sum((y-theory)**2*inv_sigma2))


# ## Set up cobaya's dict and add priors, then run

# In[18]:


info = {
    "likelihood": {
        "external": lnlike},
    "params": dict([
        ("a", {
            "prior": {"min": 5, "max": 15},
            "latex": r"\alpha"}),
        ("b", {
            "prior": {"min": 0, "max": 5},
            "latex": r"\beta"}),
        ("c", {
            "prior": {"min": -5, "max": 0},
            "latex": r"\gamma"}) ] ),
        
    "sampler": {
        "mcmc": {'max_tries':10000}}, 'output':'chains/'}

from cobaya.run import run

updated_info, sampler = run(info)


