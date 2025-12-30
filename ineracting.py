# fBmChemotaxis -- interacting particles

# Code developed by Gustavo Cornejo-Olea and Lucas Buvinic

# If you use fBmChemotaxis in your research, please cite:

# G. Cornejo-Olea, L. Buvinic, J. Darbon, R. Erban, A. Ravasio, and A. Matzavinos.  
# On the role of fractional Brownian motion in models of chemotaxis and stochastic gradient ascent. 
# Submitted, 2025. Preprint available on arXiv: 2511.18745.


import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt



#######################################
########### METHODS TO USE ############


def fbm(t_input, H):
    '''
    Sample fractional Brownian motion (fBm) of Hurst index H in t_input
    time-points using the Cholesky method.
    '''
    t = t_input.copy()
    N = len(t)

    # Covariance matrix of fBm on t_input
    C_N = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ti = t[i]
            tj= t[j]
            C_N[i, j] = 0.5 * (ti ** (2 * H) + tj ** (2 * H) - abs(ti - tj) ** (2 * H))
    [S, U] = npl.eig(C_N)

    # Generate standard normal random vector
    xsi = npr.normal(0, 1, N)
    SS = np.diag(np.sqrt(S))
    W = np.dot(np.dot(U, SS), xsi)
    return W


def fbm_noise(t_input, H):
    '''
    Sample increments of fBm of Hurst index H in t_input
    time-points.
    '''
    W = fbm(t_input, H)
    return np.diff(W)


def ind_chemo_term(x, x1, c1=0.25):
    '''
    Self generating cue for activated particles that reach the global optimum at x1. c1 indicates the strength of the signal.
    '''
    return c1 * (1 / (0.3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - x1) / 0.3) ** 2)


def crank_nicholson_crowd(eq_params, spatial_params, temporal_params, initial_condition, f, df, thr=3.5, H=0.5, xi=1, sigma=1.5):
    '''
    Simulate the motion of particles driven by a chemoattractant function f (with gradient gradf) and fBm of Hurst 
    index H, considering the diffussing self-generated signals emmited by activated cells.
    For this use a Crank-Nicholson-type method with zero bounday conditions.
    '''

    # Diffusion constant of the self-generated signals
    alpha = eq_params['alpha']

    # Boundary dimensions
    L1 = spatial_params['L1']; L2 = spatial_params['L2']

    # Number of grid-points for finite differences
    nx = spatial_params['nx']
    dx = (L2 - L1) / (nx - 1)

    # Maximum simulation time
    T = temporal_params['T']

    # Number of time-steps
    nt = temporal_params['nt']
    dt = T / nt

    # Initial condition of the particles
    starting_points = spatial_params['start']

    # Paths of the particles
    hist = [[starting_points[i]] for i in range(len(starting_points))]

    x = np.linspace(L1, L2, nx)

    # Secondary diffusible signal
    h = np.zeros((nt + 1, nx))

    # State of activation of the particles (1 if activated, 0 if not). All particles are initialized as non-activated.
    Particles_states = [0 for _ in range(len(starting_points))]

    h[0, :] = initial_condition(x)

    # Sample increments of fBm
    noises = [fbm_noise(np.arange(0, nt + 1) * dt / 2, H) for _ in range(len(starting_points))]

    # Initialize the matrices for the Crank-Nicholson scheme
    A = np.zeros((nx - 2, nx - 2))
    B = np.zeros((nx - 2, nx - 2))
    r = alpha * dt / (2 * (dx ** 2))
    for i in range(nx - 2):
        A[i, i] = 1 + r
        B[i, i] = 1 - r
        if i > 0:
            A[i, i - 1] = -r / 2
            B[i, i - 1] = r / 2
        if i < nx - 3:
            A[i, i + 1] = -r / 2
            B[i, i + 1] = r / 2

    for n in range(0, nt):

        # Include the contribution of each generated signal
        g_n = np.sum([ind_chemo_term(x[1:-1], hist[i][-1]) * Particles_states[i] for i in range(len(starting_points))], axis=0)
        G = dt * (g_n)

        b = np.dot(B, h[n, 1:-1]) + G

        # Solve linear system: A * h^{n+1} = b
        h_next = np.linalg.solve(A, b)

        # Update secondary diffusible signal function (and its gradient)
        h[n + 1, 1:-1] = h_next
        gradient_h = np.gradient(h[n + 1,:])

        # Gradient of the combined chemoattractant function
        gradient_tot = lambda y: np.interp(y, x, gradient_h) + df(y)

        # Update the position with an Euler-type scheme
        for i in range(0,len(starting_points)):
          hist[i].append(hist[i][-1] + xi * gradient_tot(hist[i][-1]) * dt / 2 + sigma * noises[i][n])

          # Set the particle state as activated if it reaches the threshold
          if f(hist[i][-1]) > thr:
            Particles_states[i] = 1
          else:
            Particles_states[i] = 0

    return x, h, hist


############ SET PARAMETERS ##########

# Hurst index of fBm
H = 0.95

# Chemotactic strength
xi = 1
# Diffusion constant of SDE
sigma = 1.5
# Stopping threshold
threshold = 3.5


###### PROBLEM SET-UP #######


# Primary chemotactic signal (strong trap) and its gradient
f_S = lambda x: 4 * np.e ** (-2 * x ** 2) + np.e ** (-(x - 3) ** 2) + np.e ** (-(x - 6) ** 2) + np.e ** (-(x - 9) ** 2)
df_S = lambda x: -16 * np.e ** (-2 * x ** 2) * x - 2 * (x - 3) * np.e ** (-(x - 3) ** 2) - 2 * (x - 6) * np.e ** (-(x - 6) ** 2) - 2 * (x - 9) * np.e ** (-(x - 9) ** 2)

# Initial condition of the diffusible signal
def initial_condition(x):
   return 0

# Initial conditions of the particle positions
start = [12] * 100

# Diffusion constant of the self-generated signals
eq_params = {'alpha': 1}
# Spatial parameters
spatial_params = {'L1': -10.0,'L2':10, 'nx': 150,'start': start}
# Temporal parameters
temporal_params = {'T': 20, 'nt': 200}


##############################
#######  SIMULATION   ########

x, h, hist = crank_nicholson_crowd(eq_params, spatial_params, temporal_params, initial_condition, thr=threshold, f=f_S, df=df_S, H=H, xi=xi, sigma=sigma)



#############################
######## PLOTING  ###########

# Number of particles that reach the threshold
particles = 0
for k in range(len(hist)):
    plt.scatter(hist[k][-1], np.interp(hist[k][-1], x, h[-1,:]) + f_S(hist[k][-1]), marker=".", s = 100)
    if f_S(hist[k][-1]) > threshold:
        particles += 1

x_plot = np.linspace(-10, 15, 200)
total_chem = np.interp(x_plot, x, h[-1, :]) + f_S(x_plot)
plt.plot(x_plot, total_chem)
plt.xlim(-10, 15)
plt.ylim(-0.5, )
plt.grid(True)
plt.title(f"H={H}, χ={xi}, σ={sigma}", fontsize=16)
plt.text(7.5, (9 / 10.6) * np.max(total_chem), particles, fontsize=22, bbox=dict(facecolor='red', alpha=0.5))

plt.show()