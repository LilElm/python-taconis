# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft
from chart import InteractivePlot, InteractiveScatter, Chart2D
from lorentzian import (fit_lorentzian, fit_noise, fit_noise_v2,
                        lorentzian_real, lorentzian_imag,
                        mag)
#from scipy.signal import butter, sosfiltfilt


class SweepFile():
    def __init__(self,
                 path,
                 folder,
                 filename,
                 date,
                 time,
                 pressure_start,
                 pressure_end,
                 heater,
                 rot,
                 roots,
                 pressurising,
                 hose,
                 reservoir,
                 freq,
                 x,
                 y,
                 label):
        self.path = path
        self.folder = folder
        self.filename = filename
        self.date = date
        self.time = time
        self.pressure_start = pressure_start
        self.pressure_end = pressure_end
        self.heater = heater
        self.rot = rot
        self.roots = roots
        self.pressurising = pressurising
        self.hose = hose
        self.reservoir = reservoir
        self.freq = freq
        self.x = x
        self.y = y
        self.label = label
        



def main():
    sweepDict = {}
    parent_dir = "C:/Users/ultservi/Desktop/Elmy/python-taconis/dat/"
    
    for folder, dirs, files in os.walk(parent_dir, topdown=False):
        if "2024.08.23" in folder:
        
        #for folder in dirs:
         #   for root2, dirs, files in os.walk(os.path.join(parent_dir, folder), topdown=False):
                for name in files:
                    # Read LC sweep data
                    if "gas" in name:
                        if "rot" in name or "root" in name:
                            if "full" in name:
                                #pressure = float(name.split("_")[3].split("mBar")[0])# + " mBar"
                                #with open((os.path.join(parent_dir, folder, root2, name)), "r") as f:
                                
                                path = os.path.join(parent_dir, folder, name)
                                """
                                print(f"===================================")
                                print(f"{folder}")
                                print("======================================\n")
                                """
                                #with open((os.path.join(parent_dir, folder, name)), "r") as f:
                                with open(path, "r") as f:
                                    headers = f.readline().strip("\n").split("\t")
                                    date = headers[4]
                                    time = headers[5]
                                    pressure_start = float(headers[6])
                                    pressure_end = float(headers[7])
                                    if "on" in headers[8]:
                                        heater = True
                                    elif "off" in headers[8]:
                                        heater = False
                                    else:
                                        heater = None
                                        
                                    if "hose" in name:
                                        hose = True
                                    else:
                                        hose = False 
                                    
                                    if "reservoir" in name:
                                        reservoir = True
                                    else:
                                        reservoir = False
                                        
                                    
                                    
                                    if "rot" in name:
                                        rot = True
                                    else:
                                        rot = False
                                        
                                    if "roots" in name:
                                        roots = True
                                    else:
                                        roots = False
                                    
                                    if "pressurising" in name:
                                        pressurising = True
                                    else:
                                        pressurising = False
                                    
                                    #data = np.loadtxt(f, dtype=str, skiprows=1, delimiter="\t", usecols=range(3))
                                    data = np.genfromtxt(f, dtype=None, delimiter="\t", usecols=range(3))
                                
                                
                                i = 0
                                for row in data:
                                    try:
                                        for elem in row:
                                            float(elem)
                                    except:
                                        data = np.delete(data, (i), axis=0)
                                    i += 1
                                
                                # Truncate first three points
                                freq = data[3:,0]
                                x = data[3:,1]
                                y = data[3:,2]
                                
                                
                                
                                label = (f"{pressure_start}-{pressure_end} mbar\n"
                                         f"rot: {rot}\n"
                                         f"roots: {roots}\n"
                                         f"pressurising: {pressurising}\n"
                                         f"heater: {heater}\n"
                                         f"reservoir: {reservoir}\n"
                                         f"hose: {hose}")
                                
                                
                                sweepDict[name] = SweepFile(path=path,
                                                            folder=folder,
                                                            filename=name,
                                                            date=date,
                                                            time=time,
                                                            pressure_start=pressure_start,
                                                            pressure_end=pressure_end,
                                                            heater=heater,
                                                            rot=rot,
                                                            roots=roots,
                                                            pressurising=pressurising,
                                                            hose=hose,
                                                            reservoir=reservoir,
                                                            freq=freq,
                                                            x=x,
                                                            y=y,
                                                            label=label)
    
    
    # Initiate Charts
    chart_sweeps = InteractivePlot(name="Frequency Sweeps at Varying Pressures")
    chart_fit = InteractivePlot(name="Fit")
    chart_fit_delta_x = InteractivePlot(name="X (Fit - Val)")
    chart_fit_delta_y = InteractivePlot(name="Y (Fit - Val)")
    chart_fit_delta_moving_x = InteractivePlot(name="X Moving Average (Fit - Val)")
    chart_fit_delta_moving_y = InteractivePlot(name="Y Moving Average (Fit - Val)")

    chart_noise_x = InteractivePlot(name="Noise\n Abs(Moving Average (X Abs(Fit - Val)) -  X Abs(Fit - Val))")
    chart_noise_y = InteractivePlot(name="Noise\n Abs(Moving Average (Y Abs(Fit - Val)) -  Y Abs(Fit - Val))")

    chart_noise_moving_x = InteractivePlot(name="Moving Average of Noise (X)")
    chart_noise_moving_y = InteractivePlot(name="Moving Average of Noise (Y)")
    

    #chart_noise_x_fab = Chart(name="Fabricated data (complete guess)")
    #chart_noise_y_fab = Chart(name="Fabricated data (complete guess)")
    
    #chart_noisy_fit_x = Chart(name="Noisy fit")
    #chart_noisy_fit_y = Chart(name="Noisy fit")
    
    
    #chart_norm = InteractivePlot(name="Lorentzians / Fit")
    

    
    #labels = []
    # Analyse data in each sweep
    for name in sweepDict:
        #input(str(name)+"============")
        label = sweepDict[name].label
        #=================================================================
        #=================================================================
        # Record raw Lorentzian data
        # sweepDict[name].x
        # chart_sweeps
        #=================================================================
        
        pressure = 0.5 * (sweepDict[name].pressure_start + sweepDict[name].pressure_end)
        sweepDict[name].pressure = pressure
        """
        label = (f"{sweepDict[name].pressure_start}-{sweepDict[name].pressure_end} mbar\n "
                 f"rot: {sweepDict[name].rot}"
                 f"roots: {sweepDict[name].roots}"
                 f"pressurising: {sweepDict[name].pressurising}"
                 f"heater: {sweepDict[name].heater}")
        labels.append(label)
        """
        chart_sweeps.add_line(sweepDict[name].freq,
                              sweepDict[name].x,
                              pressure,
                              label=label)
        chart_sweeps.add_line(sweepDict[name].freq,
                              sweepDict[name].y,
                              pressure,
                              label=label)

        
        #=====================================================================
        # Perform a Fourier transform on the Lorentzian signal
        
        x_fft = fft(sweepDict[name].x)
        y_fft = fft(sweepDict[name].y)
        n = len(sweepDict[name].freq)
        freq_fft = fftfreq(n, sweepDict[name].freq[1] - sweepDict[name].freq[0])
        
        
        x_fft_shifted = np.abs(fftshift(x_fft))
        y_fft_shifted = np.abs(fftshift(y_fft))
        #x_fft_shifted = fftshift(x_fft)
        #y_fft_shifted = fftshift(y_fft)
        freq_fft_shifted = fftshift(freq_fft)
        
        # Implement a low-pass filter
        cutoff = 0.12 # Hz
        x_fft_filtered = np.where(np.abs(freq_fft) < cutoff, x_fft, 0)
        y_fft_filtered = np.where(np.abs(freq_fft) < cutoff, y_fft, 0)
        
        x_filtered = ifft(x_fft_filtered)
        y_filtered = ifft(y_fft_filtered)
        
        
        
        
        
        
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(3,1,1)
        ax1.plot(sweepDict[name].freq, sweepDict[name].x, label="x")
        ax1.plot(sweepDict[name].freq, sweepDict[name].y, label="y")
        ax1.legend()
        
        
        ax2 = fig.add_subplot(3,1,2)
        ax2.plot(sweepDict[name].freq, x_filtered, label="filtered x")
        ax2.plot(sweepDict[name].freq, y_filtered, label="filtered y")
        ax2.legend()
        
        
        ax3 = fig.add_subplot(3,1,3)
        ax3.plot(freq_fft_shifted, x_fft_shifted, label="FFT x")
        ax3.plot(freq_fft_shifted, y_fft_shifted, label="FFT y")
        ax3.set_xbound(0,)
        ax3.legend()
        
        
        
        plt.show()
        input("00000000")
        plt.close()
        

 
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.plot(sweepDict[name].freqs_shifted, sweepDict[name].x_fft_shifted, label="x")
        plt.plot(sweepDict[name].freqs_shifted, sweepDict[name].y_fft_shifted, label="y")
        ax.legend()
        plt.show()
        input("00000000")
        plt.close()
        

        """






        #=================================================================
        #=================================================================
        # Fit Lorentzians to data
        # sweepDict[name].x_fit
        # chart_fit
        #=================================================================
        
        popt = fit_lorentzian(sweepDict[name].freq,
                              sweepDict[name].x,
                              sweepDict[name].y)
        
        w_0, vmax, dw, a, b, c, d, phase = popt
        sweepDict[name].x_fit = lorentzian_real(sweepDict[name].freq, w_0,
                                                vmax, dw,
                                                a, b,
                                                phase)
        
        sweepDict[name].y_fit = lorentzian_imag(sweepDict[name].freq, w_0,
                                                vmax, dw,
                                                c, d,
                                                phase)
        
        chart_fit.add_line(sweepDict[name].freq,
                           sweepDict[name].x_fit,
                           pressure,
                           label=label)
        
        chart_fit.add_line(sweepDict[name].freq,
                           sweepDict[name].y_fit,
                           pressure,
                           label=label)
        
        """
        #=====================================================================
        # Perform a Fourier transform on the fitted Lorentzian signal
        x_fit_fft = fft(sweepDict[name].x_fit)
        y_fit_fft = fft(sweepDict[name].y_fit)
        
        n = len(sweepDict[name].freq)
        x_fit_freqs = fftfreq(n, sweepDict[name].x_fit[1] - sweepDict[name].x_fit[0])
        y_fit_freqs = fftfreq(n, sweepDict[name].y_fit[1] - sweepDict[name].y_fit[0])
        
        
        sweepDict[name].x_fit_fft_shifted = np.abs(fftshift(x_fit_fft))
        sweepDict[name].y_fit_fft_shifted = np.abs(fftshift(y_fit_fft))
        
        sweepDict[name].x_fit_freqs_shifted = fftshift(x_fit_freqs)
        sweepDict[name].y_fit_freqs_shifted = fftshift(y_fit_freqs)
        
        
        #=====================================================================
        # Fit Fourier transforms
        """
        
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.plot(sweepDict[name].x_freqs_shifted, sweepDict[name].x_fft_shifted, label="x raw")
        plt.plot(sweepDict[name].y_freqs_shifted, sweepDict[name].y_fft_shifted, label="y raw")
        plt.plot(sweepDict[name].x_fit_freqs_shifted, sweepDict[name].x_fit_fft_shifted, label="x fit")
        plt.plot(sweepDict[name].y_fit_freqs_shifted, sweepDict[name].y_fit_fft_shifted, label="y fit")
        ax.legend()
        plt.show()
        input("00000000")
        plt.close()
        """
        
        
        #=====================================================================
        #=====================================================================
        # Divide the Lorentzian by the fit in attempt to classify the
        # elctrical noise
        
        
        #sweepDict[name].x_norm = sweepDict[name].x / sweepDict[name].x_fit
        #sweepDict[name].y_norm = sweepDict[name].y / sweepDict[name].y_fit
        """
        chart_norm.add_line(sweepDict[name].freq,
                                       sweepDict[name].x_norm,
                                       pressure,
                                       label=f"x + {label}")
        chart_norm.add_line(sweepDict[name].freq,
                                       sweepDict[name].y_norm,
                                       pressure,
                                       label=f"y + {label}")
        """
        
        """
        fig = plt.figure(layout='constrained')
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(sweepDict[name].freq, sweepDict[name].x_norm, label = "x norm")
        ax1.plot(sweepDict[name].freq, sweepDict[name].y_norm, label = "y norm")
        ax1.legend()
        
        
        ax2 = fig.add_subplot(2,1,1)
        ax2.sharex(ax1)
        ax2.plot(sweepDict[name].freq, sweepDict[name].x, label = "x")
        ax2.plot(sweepDict[name].freq, sweepDict[name].y, label = "y")
        ax2.legend()
        
        
        plt.show()
        input("=====")
        plt.close()
        """
        
        #=================================================================
        #=================================================================
        # Subtract recorded values from fit
        # sweepDict[name].x_fit_delta
        # chart_fit_delta_x
        #=================================================================
        
        sweepDict[name].x_fit_delta = (sweepDict[name].x_fit - sweepDict[name].x)
        sweepDict[name].y_fit_delta = (sweepDict[name].y_fit - sweepDict[name].y)
        
        chart_fit_delta_x.add_line(sweepDict[name].freq,
                               sweepDict[name].x_fit_delta,
                               pressure,
                               label=label)
        
        chart_fit_delta_y.add_line(sweepDict[name].freq,
                               sweepDict[name].y_fit_delta,
                               pressure,
                               label=label)



        #=====================================================================
        #=====================================================================
        # Divide the delta by the fit in attempt to classify the
        # elctrical noise
        
        
        #sweepDict[name].x_norm = sweepDict[name].x_fit_delta / sweepDict[name].x_fit
        #sweepDict[name].y_norm = sweepDict[name].y_fit_delta / sweepDict[name].y_fit
        """
        chart_norm.add_line(sweepDict[name].freq,
                                       sweepDict[name].x_norm,
                                       pressure,
                                       label=f"x + {label}")
        chart_norm.add_line(sweepDict[name].freq,
                                       sweepDict[name].y_norm,
                                       pressure,
                                       label=f"y + {label}")
        """
        
        """
        fig = plt.figure(layout='constrained')
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(sweepDict[name].freq, sweepDict[name].x_norm, label = "x norm")
        ax1.plot(sweepDict[name].freq, sweepDict[name].y_norm, label = "y norm")
        ax1.legend()
        
        
        ax2 = fig.add_subplot(2,1,1)
        ax2.sharex(ax1)
        ax2.plot(sweepDict[name].freq, sweepDict[name].x, label = "x")
        ax2.plot(sweepDict[name].freq, sweepDict[name].y, label = "y")
        ax2.legend()
        
        
        plt.show()
        input("=====")
        plt.close()
        
        """





















        #=================================================================
        #=================================================================
        # The fit_delta curves contain both the error in the Lorentzian
        # fits and the noise. Take moving averages of these and subtract
        # the fit_delta values to get just the noise.
        # sweepDict[name].x_fit_delta_moving
        # chart_fit_delta_moving_x
        #=================================================================
        
        sweepDict[name].x_fit_delta_moving = []
        sweepDict[name].y_fit_delta_moving = []
        offset = 6
        
        for i in range(len(sweepDict[name].x_fit_delta)):
            xval = np.mean(sweepDict[name].x_fit_delta[i-offset:i+offset])
            sweepDict[name].x_fit_delta_moving.append(xval)
            yval = np.mean(sweepDict[name].y_fit_delta[i-offset:i+offset])
            sweepDict[name].y_fit_delta_moving.append(yval)

        chart_fit_delta_moving_x.add_line(sweepDict[name].freq,
                             sweepDict[name].x_fit_delta_moving,
                             pressure,
                             label=label)
        
        chart_fit_delta_moving_y.add_line(sweepDict[name].freq,
                              sweepDict[name].y_fit_delta_moving,
                              pressure,
                              label=label)
        
        #=================================================================
        #=================================================================
        # The above curves contain the moving averages. To get isolated
        # noise values, find the difference between _fit_delta and
        # fit_delta_moving.
        # sweepDict[name].x_noise
        # chart_noise
        #=================================================================     
        
        sweepDict[name].x_noise = abs(sweepDict[name].x_fit_delta_moving - sweepDict[name].x_fit_delta)
        sweepDict[name].y_noise = abs(sweepDict[name].y_fit_delta_moving - sweepDict[name].y_fit_delta)

        chart_noise_x.add_line(sweepDict[name].freq,
                               sweepDict[name].x_noise,
                               pressure,
                               label=label)
        
        chart_noise_y.add_line(sweepDict[name].freq,
                               sweepDict[name].y_noise,
                               pressure,
                               label=label)
        
        #=================================================================
        #=================================================================
        # The above curves contain the 'isolated' noise values. To
        # describe the noise, find the moving averages.
        # sweepDict[name].x_noise_moving
        # chart_noise_moving
        #=================================================================

        #chart_noise_moving_x = Chart(name="Moving Average of Noise (X)")
        #chart_noise_moving_y = Chart(name="Moving Average of Noise (Y)")


        sweepDict[name].x_noise_moving = []
        sweepDict[name].y_noise_moving = []
        offset = 10
        
        for i in range(len(sweepDict[name].x_noise)):
            xval = np.mean(sweepDict[name].x_noise[i-offset:i+offset])
            sweepDict[name].x_noise_moving.append(xval)
            yval = np.mean(sweepDict[name].y_noise[i-offset:i+offset])
            sweepDict[name].y_noise_moving.append(yval)

        chart_noise_moving_x.add_line(sweepDict[name].freq,
                                      sweepDict[name].x_noise_moving,
                                      pressure,
                                      label=label)
        
        chart_noise_moving_y.add_line(sweepDict[name].freq,
                                      sweepDict[name].y_noise_moving,
                                      pressure,
                                      label=label)

        #chart_noise_moving_x.show()
        #chart_noise_moving_y.show()
        #input("00000")
        
        #=================================================================
        #=================================================================
        # There are two immediate ways I can think of to try to describe 
        # the noise. Firstly, the areas beneath the noise curves can be
        # evaluated to give a bulk value of the noise. Secondly, functions
        # can be fitted to give parameters that describe the noise
        # statistically. We shall start with the bulk value of the noise.
        # sweepDict[name].x_noise_bulk
        # chart_noise_bulk
        #=================================================================
        
        sweepDict[name].x_simpsons = simpsons(sweepDict[name].freq,
                                              sweepDict[name].x_noise_moving)
        sweepDict[name].y_simpsons = simpsons(sweepDict[name].freq,
                                              sweepDict[name].y_noise_moving)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        """
        # Try to fit Lorentzians again, but with a noise parameter
        noise_x = fit_noise_v2(sweepDict[name].freq, w_0,
                               sweepDict[name].x_fit, sweepDict[name].x)
        noise_y = fit_noise_v2(sweepDict[name].freq, w_0,
                               sweepDict[name].y_fit, sweepDict[name].y)
        
        
        
        
        x_noisy_fit, _ = fabricate_lorentzian_data(
                                      sweepDict[name].freq, w_0,
                                      vmax, dw,
                                      a, b,
                                      c, d,
                                      phase,
                                      noise_x)
                                      
                                      
                                      
        _, y_noisy_fit = fabricate_lorentzian_data(
                                      sweepDict[name].freq, w_0,
                                      vmax, dw,
                                      a, b,
                                      c, d,
                                      phase,
                                      noise_y)
        
        
        
        
        
        
        chart_noisy_fit_x.add_line(sweepDict[name].freq,
                                   x_noisy_fit,
                                   pressure,
                                   label=label)
        
        
        chart_noisy_fit_y.add_line(sweepDict[name].freq,
                                   y_noisy_fit,
                                   pressure,
                                   label=label)
        
        
        
        
        
        
        
        
        """
        """
        x_fab, y_fab = fabricate_lorentzian_data(sweepDict[name].freq, w_0,
                                                 vmax, dw,
                                                 a, b,
                                                 c, d,
                                                 phase)
        
        
        
        
        
        
        
        chart_noise_x_fab.add_line(sweepDict[name].freq,
                                   x_fab,
                                   pressure,
                                   label=label)
        
        chart_noise_y_fab.add_line(sweepDict[name].freq,
                                   y_fab,
                                   pressure,
                                   label=label)

        """
    
    chart_sweeps.show()
    #chart_noise_x.show()
    #chart_noise_y.show()
    chart_fit.show()
    
    chart_fit_delta_x.show()
    chart_fit_delta_y.show()
    chart_fit_delta_moving_x.show()
    chart_fit_delta_moving_y.show()
    
    chart_noise_x.show()
    chart_noise_y.show()
    
    chart_noise_moving_x.show()
    chart_noise_moving_y.show()
    
   # chart_norm.show()
    
    
    #=========================================================================
    # Noise scatter plot
    


    labels_no_reservoir = []
    labels_reservoir = []
    labels_hose = []
    

    pressure_no_reservoir = []
    pressure_reservoir = []
    pressure_hose = []
    
    x_no_reservoir = []
    y_no_reservoir = []
    x_reservoir = []
    y_reservoir = []    
    x_hose = []
    y_hose = []



    for name in sweepDict:
        folder = sweepDict[name].folder
        if sweepDict[name].reservoir:
            pressure_reservoir.append(sweepDict[name].pressure)
            x_reservoir.append(sweepDict[name].x_simpsons)
            y_reservoir.append(sweepDict[name].y_simpsons)
            labels_reservoir.append(sweepDict[name].label)
            
            
        elif sweepDict[name].hose:
            pressure_hose.append(sweepDict[name].pressure)
            x_hose.append(sweepDict[name].x_simpsons)
            y_hose.append(sweepDict[name].y_simpsons)
            labels_hose.append(sweepDict[name].label)
            
            
        else:
            pressure_no_reservoir.append(sweepDict[name].pressure)
            x_no_reservoir.append(sweepDict[name].x_simpsons)
            y_no_reservoir.append(sweepDict[name].y_simpsons)
            labels_no_reservoir.append(sweepDict[name].label)
        
    
    zipped_reservoir = sorted(zip(pressure_reservoir, x_reservoir, y_reservoir, labels_reservoir))
    pressure_reservoir = [x for x, _, _, _ in zipped_reservoir]
    x_reservoir = [x for _, x, _, _ in zipped_reservoir]
    y_reservoir = [x for _, _, x, _ in zipped_reservoir]
    labels_reservoir = [x for _, _, _, x in zipped_reservoir]
    
    zipped_no_reservoir = sorted(zip(pressure_no_reservoir, x_no_reservoir, y_no_reservoir, labels_no_reservoir))
    pressure_no_reservoir = [x for x, _, _, _ in zipped_no_reservoir]
    x_no_reservoir = [x for _, x, _, _ in zipped_no_reservoir]
    y_no_reservoir = [x for _, _, x, _ in zipped_no_reservoir]
    labels_no_reservoir = [x for _, _, _, x in zipped_no_reservoir]
    
    
    
    
    zipped_hose = sorted(zip(pressure_hose, x_hose, y_hose, labels_hose))
    pressure_hose = [x for x, _, _, _ in zipped_hose]
    x_hose = [x for _, x, _, _ in zipped_hose]
    y_hose = [x for _, _, x, _ in zipped_hose]
    labels_hose = [x for _, _, _, x in zipped_hose]

    
    
    
    
    labels = []
    for name in sweepDict:
        labels.append(sweepDict[name].label)
        #labels.append(sweepDict[name].label)
    
    
    
    
    #chart_noise = InteractiveScatter(name="Noise")
    chart_noise = Chart2D(name="Noise as a Function of Pressure")
    
    
    
    
    #labels = [(x,x) for x in labels]
    
    
    #chart_noise.labels = labels
    
    
    """
    chart_noise.scatter(pressure_reservoir, x_reservoir,
                        name="X Reservoir", labels=pressure_reservoir,
                        marker=".", s=64)
    
    
    chart_noise.scatter(pressure_reservoir, y_reservoir,
                        name="Y Reservoir", labels=pressure_reservoir,
                        marker=".", s=64)
        
    chart_noise.scatter(pressure_no_reservoir, x_no_reservoir,
                        name="X No Damper", labels=pressure_no_reservoir,
                        marker=".", s=64)
    
    chart_noise.scatter(pressure_no_reservoir, y_no_reservoir,
                        name="Y No Damper", labels=pressure_no_reservoir,
                        marker=".", s=64)
    
    
    chart_noise.scatter(pressure_hose, x_hose,
                        name="X Hose", labels=pressure_hose,
                        marker=".", s=64)
    chart_noise.scatter(pressure_hose, y_hose,
                        name="Y Hose", labels=pressure_hose,
                        marker=".", s=64)
    """
    chart_noise.scatter(pressure_reservoir, x_reservoir,
                        name="X Reservoir", labels=labels_reservoir,
                        marker=".", s=64)
    
    
    chart_noise.scatter(pressure_reservoir, y_reservoir,
                        name="Y Reservoir", labels=labels_reservoir,
                        marker=".", s=64)
        
    chart_noise.scatter(pressure_no_reservoir, x_no_reservoir,
                        name="X No Damper", labels=labels_no_reservoir,
                        marker=".", s=64)
    
    chart_noise.scatter(pressure_no_reservoir, y_no_reservoir,
                        name="Y No Damper", labels=labels_no_reservoir,
                        marker=".", s=64)
    
    
    chart_noise.scatter(pressure_hose, x_hose,
                        name="X Hose", labels=labels_hose,
                        marker=".", s=64)
    chart_noise.scatter(pressure_hose, y_hose,
                        name="Y Hose", labels=labels_hose,
                        marker=".", s=64)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

    
    
    
    chart_noise.show()
    input("======")
    chart_noise.close()
    
    
    
    
    
    
    
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    
    ax.set_title(folder)
    
    ax.scatter(pressure_reservoir,
            x_reservoir,
            label="x reservoir")
    
    
    ax.scatter(pressure_reservoir,
            y_reservoir,
            label="y reservoir")
        
        
    ax.scatter(pressure_no_reservoir,
            x_no_reservoir,
            label="x no reservoir")        
    ax.scatter(pressure_no_reservoir,
            y_no_reservoir,
            label="y no reservoir")
    
    
    ax.scatter(pressure_hose,
            x_hose,
            label="x hose")        
    
    ax.scatter(pressure_hose,
            y_hose,
            label="y hose")
        
        

    
    
    
    
    ax.legend()
    plt.show()
    
    
    
    
    input("===")



    """
    
        

    
    
    
        
        




    





def fabricate_lorentzian_data(w, w_0, vmax, dw, a, b, c, d, phase, noise=0.00005):
    # Fabricate noisy Lorentzian data
    
    x = lorentzian_real(w, w_0, vmax, dw, a, b, phase)
    y = lorentzian_imag(w, w_0, vmax, dw, c, d, phase)
    
    noise_x = np.random.normal(0, noise, len(x))
    noise_y = np.random.normal(0, noise, len(y))
    x = x + noise_x
    y = y + noise_y
    #r = list(map(mag, (x), (y)))

    return x, y
    




#def simpsons(b, a, n, fn, k):
def simpsons(x, y):
    x = np.array(x)
    y = np.array(y)
    
    
    mask = (np.isnan(y)==False)
    #print(f"mask = {mask}")
    
    x = x[mask]
    y = y[mask]
    
    #print(str(x))
    #print(str(y))
    #y = y[~np.isnan(y)]
    
    
    n = len(x)
    h = (x[-1] - x[0]) / (n-1)
    
 #   print(f"h = {h}")
  #  print(f"y = {y}")
    
   # print(f"{(y[0] + 4*np.sum(y[1:-1:2]))}")
   # print(f"{2*np.sum(y[2:-1:2]) + y[-1]}")
   # print(f"{y[0]}")
   # print("=================\n")
    
    
    #y = fn(x, k)
    #return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])






if __name__ == "__main__":
    main()