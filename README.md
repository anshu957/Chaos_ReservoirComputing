## A very basic implementation of Reservoir that can learn chaotic systems

### To run you need to have the following installed in your working python environment:
1. Numpy 
2. [Jitcode](https://github.com/neurophysik/jitcode) 
3. [NetworkX](https://networkx.github.io/)

### Example(Rossler Attractor): 

1. __To run__:  python3 lorenz.py 
2. --> Command line arguments can be given. For complete list of arguments, see
**lorenz.py**
3. --> Warning: Parameters, **leaky_rate** and **inputScaling_radius** are quite sensitive for reasons currently unknown to me. Avoid changing them! 

4. --> At the moment, **rossler** is not behaving normally. Since every compilation produces different random numbers for weights, sometimes the reservoir output blows up  and sometimes it learns the system quite well. Parameters for rossler at not great right now and its better to avoid it.    
