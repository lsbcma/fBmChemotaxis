# fBmChemotaxis

fBmChemotaxis is a Python-based simulation tool for simulating chemotaxis with autocorrelated noise. The project implents a simple model of chemotaxis where cell motillity is driven by a combination of the gradient of a chemoattractant and fractional Brownian motion (fBm), represented by a Stochastic Differential Equation (SDE). 
The code allows for the simulation of the increments of fBm via the Davies and Harte method and Cholesky method [1], and the solving of the SDE by a Euler-type numerical scheme. Aditionally, the equations can be solved on a manifold in local coordinates when a global chart is defined. 
(Text about interacting particles)

[1]: Citation Dieker

## Features
- Simulation of chemotaxis model.
- Is solved on a manifold.
- Incorporates a simple model of communication between cells.

## Installation
### Prerequisites
- A Python compiler.
- `numpy` and `matpotlib`.

### Compile the Code
For 2D chemotaxis on a manifold run the `manifold.py` file. To simulate 2D chemotaxis on Euclidan space, set the manifold parameters to 0.

For 1D chemotaxis with interacting particles run the `interacting.py` file.

## File Structure
- `manifold.py`: Code for simulating 2D chemotaxis on a manifold (or Euclidean space).
- `interacting.py`: Code for simulating 1D chemotaxis with interacting particles.
- `README.md`: Documentation for the repository.

## Citation
If you use fBmChemotaxis in your research, please cite:

Paper citation