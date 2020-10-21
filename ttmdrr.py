import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import *
import pylab
import math

# main program
if __name__ == "__main__":
    global n,nx,EYE,tau,eps,rma,dim,ro,po,rd
    global dx,dp,m,om,nsc,Pxy,gam,d
    global ttzeros

    eshift=2.48
    nsc=400
    tau=25
    dw=2.0*np.pi/(tau*10*nsc)
    EYE = complex(0,1)  # imaginary number 

    qrt = np.zeros((nsc*10),dtype=complex)  
    crt = np.zeros((nsc*10),dtype=complex)  
    rw = np.zeros((nsc*10),dtype=complex)  
# read parameters
    th= loadtxt('times.npy')
#    qrh= loadtxt('qrt_real.npy')
#    qih= loadtxt('qrt_imag.npy')
    crh= loadtxt('crt_real.npy')
    cih= loadtxt('crt_imag.npy')

#    qpt0= loadtxt('qppt0.npy')
#    qpc1= loadtxt('qppc1.npy')
    cpc1= loadtxt('cppc1.npy')
    cpt0= loadtxt('cppt0.npy')
    cpc0= loadtxt('cppc0.npy')
    cpt1= loadtxt('cppt1.npy')

    for i in range(nsc):
#        qrt[i]=qrh[i]+EYE*qih[i]
        crt[i]=crh[i]+EYE*cih[i]

    ax= plt.subplot(3,1,1)
#    ax.plot(th,np.real(qrt[:nsc]))
#    ax.plot(th,np.imag(qrt[:nsc]))
#    ax.plot(th,np.abs(qrt[:nsc]))
    ax.plot(th,np.real(crt[:nsc]))
    ax.plot(th,np.imag(crt[:nsc]))
    ax.plot(th,np.abs(crt[:nsc]))

    ax= plt.subplot(3,1,2)
#    ax.plot(th,qpt0)
#    ax.plot(th,qpc1)
    ax.plot(th,cpt0)
    ax.plot(th,cpc1)
    ax.plot(th,cpt1)
    ax.plot(th,cpc0)

    w=np.fft.fftshift(np.fft.fftfreq(nsc*10,1.0/(10*nsc*dw)))
#    rw=np.fft.fftshift(np.fft.ifft(qrt))
    rw=np.fft.fftshift(np.fft.ifft(crt))
    ax= plt.subplot(3,1,3)
#    print(np.size(w))
#    print(np.size(rw))
    ax.plot(w*27.2+eshift,np.real(rw))
    ax.set_xlim(2,3)
    ax.grid()

    plt.pause(15.)

