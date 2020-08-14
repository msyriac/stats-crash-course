# ## Plot with getdist

# In[19]:


# Export the results to GetDist

from getdist.mcsamples import loadMCSamples
import matplotlib.pyplot as plt
gd_sample = loadMCSamples("chains/")

# Analyze and plot
mean = gd_sample.getMeans()
covmat = gd_sample.getCovMat().matrix
print("Mean:")
print(mean)
print("Covariance matrix:")
print(covmat)
# %matplotlib inline  # uncomment if running from the Jupyter notebook
import getdist.plots as gdplt

true_a = 10
true_b = 2
true_c = -3

gdplot = gdplt.get_subplot_plotter()
gdplot.triangle_plot(gd_sample, ["a", "b","c"], filled=True,markers={"a":true_a, "b":true_b,"c":true_c})
plt.savefig('contours.png')

# ##  MPI
# 
# MPI parallelization is easy. Simply specify an `output` directory in the dictionary so that the chains will be written to disk. Then call the script with the cobaya run command in it with mpirun (or whatever MPI wrapper you use) and the number of chains will equal the number of MPI processes. Then load the chains using getdist and plot them.
