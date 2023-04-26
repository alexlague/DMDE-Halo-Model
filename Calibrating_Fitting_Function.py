####
# Spherical Collapse with kinetically interacting DMDE
#
# GENERAL PROCEDURE:
#
# Integrate from delta_ini = delta(a_ini) until today
# Vary delta_ini until the collapse redshift is z_coll=0
# Find the corresponding value of the LINEAR delta at z=0
# using the linear growth factor
#
#
####


import numpy as np
from scipy.integrate import odeint, solve_ivp, quad
from scipy.interpolate import CubicSpline

def compute_linear_g(Omegam, w, Gamma_over_rho):
    # Solving for unnormalized growth factor
    # H0 factors out except for Gamma which is Gamma/H0/rho_c
    # Initial conditions are g = a for a << 1 and g' = 1

    #Omegam = 0.3
    OmegaL = 1-Omegam
    #w = -.8
    #Gamma_over_rho = 6*Omegam
    H_of_a = lambda a: np.sqrt(Omegam*a**-3 + OmegaL*a**(-3*(1+w)))
    coeff1 = lambda a: (1.5*Omegam * a**-3/H_of_a(a)**2-3+a**2/H_of_a(a)*Gamma_over_rho/Omegam)/a
    coeff2 = lambda a: 1.5 / a**2 * Omegam * a**-3 / H_of_a(a)**2

    def g_derivatives(x, y):
        return [y[1], coeff1(x)*y[1]+coeff2(x)*y[0]]


    a_ini = 1e-4
    a_array = np.linspace(a_ini, 1.0, 400)
    
    ivp_sol = solve_ivp(g_derivatives,  [a_ini, a_array[-1]], 
                        [a_ini, 1],
                        t_eval=a_array, method='RK45', rtol=1e-7)

    g_interp = CubicSpline(ivp_sol['t'], ivp_sol['y'][0])

    return g_interp

def compute_delta_c(Omegam, w, Gamma_over_rho):
    # Define cosmology
    # x = ln(a)
    
    a_ini = 1e-4
    x_ini = np.log(a_ini)
    sc_m = lambda x: np.exp(-3*x) # scalings with a
    sc_L = lambda x: np.exp(-3*(1+w)*x)

    #Compute linear growth
    g_interp = compute_linear_g(Omegam, w, Gamma_over_rho)
    OmegaL = 1-Omegam
    
    # Solving 0812.0545 Eq. (A10)
    
    coeff1 = lambda x: 1.5*(Omegam*sc_m(x)/(Omegam*sc_m(x)+OmegaL*sc_L(x)))
    coeff2 = lambda x: -0.5*(Omegam*sc_m(x)
                             +(1+3*w)*OmegaL*sc_L(x))/(Omegam*sc_m(x)+OmegaL*sc_L(x))
    coeff3 = lambda x: -0.5*Omegam*sc_m(x)/(Omegam*sc_m(x)+OmegaL*sc_L(x))
    coeff4 = lambda x: -np.exp(3*x) / np.sqrt(Omegam*sc_m(x)+OmegaL*sc_L(x)) / Omegam

    def y_derivatives(x, y):
        return [y[1], coeff1(x)*y[1]+coeff2(x)*y[0]+ 
                coeff3(x)*(np.exp(x)/np.exp(x_ini)+y[0])
                *((y[0]*np.exp(x_ini)/np.exp(x)+1)**(-3)*(1+delta_ini)-1)+
                coeff4(x)*Gamma_over_rho*(y[1]-y[0])]
    
    # Initial conditions from 1703.05824 below Eq. (14)
    x_array = np.linspace(x_ini, np.exp(1), 10000)
    i = 0
    for delta_ini in np.linspace(5e-4, 1e-4, 200):
        ivp_sol = solve_ivp(y_derivatives,  [x_ini, x_array[-1]], 
                            [0, -delta_ini/3/(1+delta_ini)],
                            t_eval=x_array, method='Radau', rtol=1e-4)
        x_array_cut = x_array[5000:len(ivp_sol['y'][0])]
        i_coll = np.argmin(abs(-np.exp(x_array_cut-x_ini)-ivp_sol['y'][0][5000:]))
        x_coll = x_array_cut[i_coll]
        a_coll = np.exp(x_coll)
        if 1/a_coll-1 <0:
            delta_c_out = np.linspace(5e-4, 1e-4, 200)[i-1] * g_interp(1.)/g_interp(a_ini)
            break
        i+=1

    return delta_c_out


def delta_c_fit(Omega_m, g, a):
    # based on Mead 2020
    
    return


# Latin hypercube for Omega_m, Gamma, and w
from scipy.stats import qmc

l_bounds = [0.1, -1, 0]
u_bounds = [1, -0.6, 8]
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=5)
sample = qmc.scale(sample, l_bounds, u_bounds)

for s in sample:
    print(compute_delta_c(*s)/1.686)
    
