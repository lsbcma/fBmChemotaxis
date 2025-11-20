# fBmChemotaxis

fBmChemotaxis is a Python-based simulation tool for simulating chemotaxis with autocorrelated noise. The project implents a simple model of chemotaxis where cell motillity is driven by a combination of the gradient of a chemoattractant and fractional Brownian motion (fBm), represented by a Stochastic Differential Equation (SDE). 
The code allows for the simulation of the increments of fBm via the Davies and Harte method and Cholesky method (in the manifold.py and interacting.py codes, respectively), and the solving of the SDE by a Euler-type numerical scheme. In the `manifold.py` code the 2D equations can be solved on a manifold in local coordinates when a global chart is defined. In the `interacting.py` code the 1D equations are solved considering the diffussing self-generated signals emmited by activated cells.

## Features
- Simulation of chemotaxis model.
- Can be solved in 2D on a manifold.
- Incorporates a simple model of communication between cells.

## Installation
### Prerequisites
- A Python compiler.
- `numpy` and `matpotlib`.

### Compile the Code
For 2D chemotaxis on a manifold run the `manifold.py` file. To simulate 2D chemotaxis on Euclidan space, set the manifold parameters to 0.

For 1D chemotaxis with interacting cells run the `interacting.py` file.

## File Structure
- `manifold.py`: Code for simulating 2D chemotaxis on a manifold (or Euclidean space).
- `interacting.py`: Code for simulating 1D chemotaxis with interacting cells.
- `README.md`: Documentation for the repository.

## Citation
If you use fBmChemotaxis in your research, please cite:

Paper citation