import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt







def fit_noise_v2(w, w0, data, tar):
    
    # Noise depends on distance away from resonant frequency
    x = abs(w0 - w)**5.0
    
    
    p0 = [2*abs(data[1] - data[0])]
    #fault = 0.0025
    
    try:
       
        
        foo = lambda independent: fit(noise_v2, independent, tar, p0)
        popt, pcov, st_err_sum = foo([x, data])
            
            
        print(f"popt (v2) = {popt}")
    except:
        return False

    if st_err_sum > 5000:
        popt = None
        print("Failed to fit.")
    else:
        try:
            print(f"popt_best = {popt_best}")
            print(f"st_err_sum = {st_err_sum}")
        except:
            print("\nFailed to fit. (v2")
            print("==============================\n\n")
            
    return popt
    
    



def fit_noise(data, tar):
    p0 = [abs(data[1] - data[0])]
    fault = 0.0025
    
    try:
        popt, pcov, st_err_sum = fit(noise, data, tar, p0)
    except:
        return False

    if st_err_sum > 5000:
        popt = None
        print("Failed to fit.")
    else:
        try:
            print(f"popt_best = {popt_best}")
            print(f"st_err_sum = {st_err_sum}")
        except:
            print("Failed to fit.")
            
    return popt
    
    

def noise(data, noise):
    noise_x = np.random.normal(0, noise, len(data))
    data = data + noise_x
    return data





def noise_v2(independent, noise):
    x, data = independent
    noise_x = np.random.normal(0, noise, len(data))
    data = data + noise_x/x
    return data




def fit_lorentzian(w, x, y):

    # Guess initial parameters for Lorentzian
    p0, tar = parameters(w, x, y)
    fault = 0.0025
     
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(w, x, label='x')
    ax.plot(w, y, label='y')
    
    
    plt.legend()
    plt.show()
    
    #fig.savefig(path, bbox_inches="tight", dpi=600)
    #plt.close()
    """
    
    
     
    try:
        popt, pcov, st_err_sum = fit(lorentzian_both, w, tar, p0)
        popt_best = popt
        if st_err_sum > fault or np.isnan(st_err_sum):
            try:
                # Try swapping x and y (in-phase and quadrature components)
                p0, tar = parameters(w, y, x)
                popt, pcov, st_err_sum_new = fit(lorentzian_both, w, tar, p0)
                if st_err_sum_new < st_err_sum:
                    st_err_sum = st_err_sum_new
                    popt_best = popt
            except:
                pass
    except:
        try:
            if popt:
                pass
        except:
            try:
                # Try swapping x and y (in-phase and quadrature components)
                p0, tar = parameters(w, y, x)
                popt_best, pcov, st_err_sum = fit(lorentzian_both, w, tar, p0)
            except:
                return False

    if st_err_sum > 5000:
        popt_best = None
        print("Failed to fit.")
    else:
        try:
            print(f"popt_best = {popt_best}")
            print(f"st_err_sum = {st_err_sum}")
        except:
            print("Failed to fit.")
            
    return popt_best
     
     
     
def fit(fn, x, tar, p0):
    popt, pcov = scipy.optimize.curve_fit(fn, x, tar, p0, sigma=1/tar, absolute_sigma=False)
    st_err = np.sqrt(np.diag(np.abs(pcov)))
    st_err_sum = sum(st_err)
    return popt, pcov, st_err_sum
     
     
     




def parameters(w, x, y):
    # Guess initial parameters for Lorentzian
    
    vmax = np.amax(y) - np.amin(y)
    vmax = 0.05
    maxloc = y.argmax(axis=0)
    minloc = y.argmin(axis=0)
    index = int(0.5 * (maxloc + minloc))
    w_0 = w[index]
    dw = np.abs(w[minloc] - w[maxloc])
    
    # Determine the phase from the gradient of the in-phase and quadrature
    # componenets of the Lorentzian
    if x[0] < x[index]:
        if minloc < maxloc:
            phase = np.pi / 4.0
        else:
            phase = 7.0 * np.pi / 4.0
    else:
        if minloc < maxloc:
            phase = 3.0 * np.pi / 4.0
        else:
            phase = 5.0 * np.pi / 4.0
    
    
    # a, b, c, d are offset terms
    a = 0.5 * (x[0] + x[-1])
    c = 0.5 * (y[0] + y[-1])
    b = (x[-1] - x[0]) / (w[-1] - w[0])
    d = (y[-1] - y[0]) / (w[-1] - w[0])
    p0 = [w_0, vmax, dw, a, b, c, d, phase]
    
    # Define target
    tar = np.concatenate((x, y))
    return p0, tar














def lorentzian_real(w, w_0, vmax, dw, a, b, phase):
    return a + b * w + (vmax * w**3.0 * dw / ((w_0**2.0 - w**2.0)**2.0 + dw**2.0 * w**2.0)) / np.cos(-1.0 * phase)


def lorentzian_imag(w, w_0, vmax, dw, c, d, phase):
    return c + d * w + vmax * w**2.0 * (w_0**2.0 - w**2.0) / ((w_0**2.0 - w**2.0)**2.0 + dw**2.0 * w**2.0) / np.sin(-1.0 * phase)


def lorentzian_both(w, w_0, vmax, dw, a, b, c, d, phase):
    real = lorentzian_real(w, w_0, vmax, dw, a, b, phase)
    imag = lorentzian_imag(w, w_0, vmax, dw, c, d, phase)
    out = np.concatenate((real, imag))
    return out


def lorentzian_both_noise(w, w_0, vmax, dw, a, b, c, d, phase, noise):
    real = lorentzian_real(w, w_0, vmax, dw, a, b, phase)
    imag = lorentzian_imag(w, w_0, vmax, dw, c, d, phase)
    
    
    noise_x = np.random.normal(0, noise, len(real))
    noise_y = np.random.normal(0, noise, len(imag))
    
    real = real + noise_x
    imag = imag + noise_y
    
    out = np.concatenate((real, imag))
    return out




def mag(a, b):
    return math.sqrt(a**2.0 + b**2.0)
