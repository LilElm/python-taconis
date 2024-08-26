### python-taconis
<sub>
  A Python program to analyse flying ball LC data in varying pressures of helium gas with the goal of identifying, quantifying and eliminating noise. The most consequential noise is thought to be from thermoacoustic (Taconis) oscillations, hence the name. Electrical noise is also prevalent and will also be addressed. The frequency sweeps being analysed are of Lorentzian form.


</sub>

#### Aims
<sub>

 * To plot Lorentzian signals

 * To isolate the noise in these signals

 * To filter out the electrical noise, leaving the thermoacoustic noise

 * To plot Fourier transforms to identify the key frequencies of the Taconis oscillations

 * To understand how Taconis noise varies with pressure

 * To minimise the Taconis noise

 * To minimise the electrical noise


</sub>

#### To Do
<sub>

   - [X] Plot Lorentzians
   - [X] Isolate the noise
   - [ ] Filter out the electrical noise
   - [X] Decompose the signal with Fourier transforms
   - [ ] Identify key frequencies of Taconis oscillations
   - [ ] Reliably minimise the Taconis noise


</sub>

#### Log
<sub>

 **26-Aug-24**

 * Created GitHub repository and uploaded previous work

 * Fourier transforms reveal that the sampling rate from 2024.08.23 data is insufficient

 * FFT low-pass filters implemented

 

</sub>