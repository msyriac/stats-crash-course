Generic MCMC with Cobaya
========================

See the notebook `cobaya_like.ipynb` for a quick tutorial/demo.

To run the same demo with MPI, you do (e.g. for 2 parallel chains):

```
mpirun -np 2 python cobaya_like.py
python cobaya_analyze.py
```