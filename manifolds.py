# fBmChemotaxis

# Code developed by Gustavo Cornejo-Olea and Lucas Buvinic

# If you use fBmChemotaxis in your research, please cite:

# Paper citation


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


#######################################
########### METHODS TO USE ############

def fbm_increments(T, N, H):
    '''
    Sample increments of fractional Brownian motion (fBm) of Hurst index H in N equidistant
    time-points from 0 to T using the Davies and Harte method [1].
    '''

    # Covariance function of fBm evaluated on the sampled time-points.
    k = np.arange(0, N, dtype=np.float64)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))

    r = np.append(gamma, 0)
    r = np.append(r, np.flip(gamma)[0:N - 1])

    # Eigenvalues of the circulant covariance matrix C
    lambd = np.flip(np.fft.fft(r * np.exp(2 * np.pi * 1j * np.arange(0, 2 * N, dtype=np.float64)
                                           * ((2 * N - 1) / (2 * N)))))

    # Generate standard normal random vectors V^(1) and V^(2)
    V1 = np.random.standard_normal(size=N + 1)
    V2 = np.random.standard_normal(size=N + 1)

    w = np.zeros(2 * N, dtype=np.complex128)   
    w[0] = V1[0] / np.sqrt(2 * N)
    w[1:N] = (V1[1:N] + 1j * V2[1:N]) / np.sqrt(4 * N)
    w[N] = V1[N] / np.sqrt(2 * N)
    w[N + 1:2 * N] = np.flip((V1[1:N] - 1j * V2[1:N]) / np.sqrt(4 * N))
    w *= np.sqrt(lambd)

    Z = np.fft.fft(w)

    return (T / N) ** H * np.real(Z[:N])


def simulate_process(T, N, H, xi, sigma, thresh, x0, y0, f, gradf, gradpsi, grad2psi):
    '''
    Simulate the motion of a particle on a manifold (with first and second-order derivatives given
    by gradpsi and grad2psi) driven by chemotaxis function f (with gradient gradf) and fBm of Hurst index H.
    The particle starts at (x0, y0) and stops when the chemotaxis function reaches thresh.
    The path of the particle is sampled at N equidistant time-points from 0 to T.
    Constants xi and sigma indicate the chemotactic strength and diffusion constant of the model.
    '''
    
    # Sample two indepent batches of fBm for each dimension
    dWH1 = fbm_increments(T, N, H); dWH2 = fbm_increments(T, N, H)

    x = [x0]; y = [y0] # Sample path of the process
    hit = 0 # 1 if the particle reaches the threshold
            # 0 if it does not

    for i in range(N):

        # Euclidean gradient of f
        gradf1, gradf2 = gradf(x[-1], y[-1])

        # First derivatives of manifold graph function
        gradpsi1, gradpsi2 = gradpsi(x[-1], y[-1])

        # Calculate the determinant and inverse of the metric
        detg = 1 + gradpsi1 ** 2 + gradpsi2 ** 2
        detginv = 1 / detg
        ginv11 = 1 - gradpsi1 ** 2 * detginv; ginv22 = 1 - gradpsi2 ** 2 * detginv; ginv12 = - gradpsi1 * gradpsi2 * detginv

        # Correction term for advection
        correcterm1 = 0; correcterm2 = 0
        if H == 0.5:
            # Second derivatives of manifold graph function
            gradpsi11, gradpsi22, gradpsi12 = grad2psi(x[-1], y[-1])
            term = -0.5 * sigma ** 2 * (gradpsi11 + gradpsi22
                    - (gradpsi1 ** 2 * gradpsi11 + gradpsi2 ** 2 * gradpsi22 
                       + gradpsi1 * gradpsi2 * gradpsi12) * detginv) * detginv
            correcterm1 = gradpsi1 * term; correcterm2 = gradpsi2 * term
        
        # Square root of the inverse metric for diffusion
        term = 1 / (detg + np.sqrt(detg))
        sqrtginv11 = 1 - gradpsi1 ** 2 * term
        sqrtginv12 = 1 - gradpsi2 ** 2 * term
        sqrtginv22 = - gradpsi1 * gradpsi2 * term

        # Update the position with an Euler-type scheme
        x.append(x[-1] + (xi * (ginv11 * gradf1 + ginv12 * gradf2) + correcterm1) * dt
                + sigma * (sqrtginv11 * dWH1[i] + sqrtginv12 * dWH2[i]))
        y.append(y[-1] + (xi * (ginv12 * gradf1 + ginv22 * gradf2) + correcterm2) * dt
                + sigma * (sqrtginv12 * dWH1[i] + sqrtginv22 * dWH2[i]))
        
        # If the particle reaches the threshold, end the simulation
        if f(x[-1], y[-1]) >= thresh:
            hit = 1
            break
    
    return x, y, hit


############ SET PARAMETERS ##########

# Maximum time for simulation
T = 10
# Number of time-steps
N = 100
# Size of the time-steps
dt = T / N

# Hurst index of fBm
H = 0.75

# Chemotactic strength
xi = 0.9
# Diffusion constant
sigma = 0.7
# Stopping threshold
thresh = 3.5


###### PROBLEM SET-UP #######

# Chemoattractant function (weak trap) with its gradient
f = lambda x, y : 4 * np.exp(-(x ** 2 + y ** 2)) + np.exp(-((np.sqrt(x ** 2 + y ** 2) - 5) ** 2 / 4))
def gradf(x, y):
    r2 = x ** 2 + y ** 2
    common_factor = -2 * np.exp(-r2)
    dx = 4 * x * common_factor; dy = 4 * y * common_factor
    r = np.sqrt(r2)
    common_factor = (5 - r) * np.exp(-(r - 5)**2 / 4) / (2 * r)
    dx2 = x * common_factor; dy2 = y * common_factor
    return (dx + dx2, dy + dy2)

# Function for the paraboloid (and its derivatives)
lambd = 0.1
psi = lambda x, y : lambd * (x ** 2 + y ** 2)
gradpsi = lambda x, y : (2 * lambd * x, 2 * lambd * y)
grad2psi = lambda x, y : (2 * lambd, 2 * lambd, 0)

x0 = 6; y0 = 6 # Starting points



##############################
#######  SIMULATION   ########


# Number of particles to simulate
N_part = 100

# Enpoint coordinates of each particle
x_end = []; y_end = []
# Total number of particles that reach the threshold
tot_hits = 0
for n in range(N_part):
    x, y, hit = simulate_process(T, N, H, xi, sigma, thresh, x0, y0, f, gradf, gradpsi, grad2psi)
    x_end.append(x[-1]); y_end.append(y[-1])
    tot_hits += hit



#############################
######## PLOTING  ###########


fig, ax = plt.subplots(1, 1)
font = {'size' : 12}
plt.rc('font', **font)

# Plot the chemoattractant function
x_plot = np.linspace(-10, 15, 500); y_plot = np.linspace(-10, 15, 500)
X, Y = np.meshgrid(x_plot, y_plot)
norm = TwoSlopeNorm(vmin=0.2, vcenter=1.2, vmax=3.5)
cmap = plt.get_cmap('jet')
im = ax.imshow(f(X, Y), cmap, extent=(x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]), origin='lower', norm=norm)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Plot the endopoints of each particle
for n in range(N_part):
    ax.scatter(x_end[n], y_end[n], edgecolors='pink')

# Show the starting point
ax.scatter(x0, y0, marker="x", s=100, c="red", linewidths=3)
ax.hlines(x0, x_plot[0], x_plot[-1])
ax.vlines(y0, y_plot[0], y_plot[-1])

# Show the number of particles that reach the threshold
ax.text(10, 10, tot_hits, bbox=dict(facecolor='red', alpha=1))

ax.set_title(f"H={H}, χ={xi}, σ={sigma}", fontsize=16)
ax.set_xlim([x_plot[0], x_plot[-1]])
ax.set_ylim([y_plot[0], y_plot[-1]])

plt.show()