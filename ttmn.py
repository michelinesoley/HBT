# ttsoft propagation with a reaction surface Hamiltonian - VSB 10/11/20
# Surfaces described by Eqs. 19-24 [J. Photochem. Photobiol. A: Chemistry 190, 274–282 (2007)].
# Corresponding TDSCF calculation [J. Phys. Chem. B  108, 6745-6749 (2004)] 

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
import tt
from numpy import linalg as LA

def parameters():
    global nstates,n,nx,EYE,tau,eps,rma,dim,rd,ro,po
    global dx,dp,nsc,gam,d,ddx
    global sig,m,om,eshift
    global wj,cj,qmodes,wfflag,Vb1,Vb2,aflag

    aflag=0                # flag for visualization of adiabatic populations
    qmodes=23              # qmodes = 0 for TDSCF simulation
    wfflag = 0             # wfflag = 1 to visualize wavepacket (if qmodes > 1, edit lines with ttpsi2[:,:,nsl])
    nsc = 400             # number of propagation steps
    tau = 25              # propagation time step
    gam = 0.001            # phenomeological dumping factor for simulating homogeneous broadening
    eps = 1e-14            # tt approx error
    rma = 3 # 5                # max tt rank
    eshift = 2.48          # energy shift for PESs
    dim = 2+qmodes         # number of coords
    nstates=2              # number of surfaces
    Vb1 = 0                # classical bath energy PES1
    Vb2 = 0                # classical bath energy PES2
    d = 8
    n = 2**d               # number or grid points
    Lx = 2*np.pi           # box size x
    Ly = 10.0              # box size y
    L = np.ones((dim))*Ly  # Box width
    L[0] = Lx
    ro=np.zeros(dim,dtype=float)       # initial wavepacket position
    po=np.ones(dim,dtype=float)*0.     # initial wavepacket momentum
    nx=np.zeros(dim,dtype=int)         # number of grid points per dimension
    dx=np.zeros(dim,dtype=float)       # grid point spacing
    dp=np.zeros(dim,dtype=float)       # momentum grid point spacing
    for i in range(dim):
        nx[i] = n                      # number of grid points
        dx[i] = L[i]/nx[i]             # coord grid spacing
        dp[i] = 2.0*np.pi/L[i]         # momenta grid spacing
    ddx=1.0
    for i in range(dim):
        ddx=ddx*dx[i]
    EYE = complex(0,1)                 # imaginary number
    m = np.ones((dim))                 # masses
    om = np.ones((dim))                # frequencies
    sig = np.ones((dim))*np.sqrt(2.0)  # Gaussian widths

    # Parameters for the first 2 modes (large amplitude modes of reaction surface, theta and x_str) 
    sig[0] = 0.15228275
    m[0] = 56198.347
    om[0] = 2.0/sig[0]**2/m[0]
    m[1] = 143.158
    om[1] = 1.0/m[1]
    return()

def mv22(emat,psi):
    # nstates x nstates matrix times vector valued tensor trains
    global eps,rma,ttzeros,nstates
    out=[]
    for j in range(nstates):
        out.append(ttzeros)
        for k in range(nstates):
            out[j] = out[j] + emat[j,k]*psi[k]
        out[j] = out[j].round(eps,rma)
    return out

def expA(A,e,eps):
    # Taylor expansion of exp(A), with A an nstates x nstates matrix valued tensor train
    global nstates,rma
    N=10
    w0=A*(1.0/2**N)
    tm=e
    k=N-1
    while k > 0:
        prod=e*0.0
        for j in range(nstates):
            for i in range(nstates):
                 for kk in range(nstates):
                     prod[j,i]=prod[j,i]+tm[j,kk]*w0[kk,i]*(1.0/k)
        tm=e+prod
        for j in range(nstates):
            for i in range(nstates):
                tm[j,i]=tm[j,i].round(eps,rma)
        k=k-1
    while k < N:
        prod=e*0.0
        for j in range(nstates):
            for i in range(nstates):
                 for kk in range(nstates):
                     prod[j,i]=prod[j,i]+tm[j,kk]*tm[kk,i]
        for j in range(nstates):
            for i in range(nstates):
                tm[j,i]=prod[j,i].round(eps,rma) 
        k=k+1
    return tm

def tt_soft(fxy,emat,Pxy):
    # soft propagation
    global nstates,eps,rma
    out=mv22(emat,fxy)
    for j in range(nstates):
        fp=mfft(out[j],1)*Pxy
        fp=fp.round(eps,rma)
        out[j]=mfft(fp,-1)
    out=mv22(emat,out)
    return out

def mfft(f,ind):
    # multidimensional fft of function f in tt format
    # ind=1 for fft, otherwise ifft
    global eps, rma
    y=f.to_list(f)                                 # get cores
    for k in range(len(y)):                        # core index
        for i in range(y[k].shape[0]):             # left index of core k
            for j in range(y[k].shape[2]):         # right index of core k
                if ind == 1:
                    y[k][i,:,j] = np.fft.fft(y[k][i,:,j])
#*4/n
                else:
                    y[k][i,:,j] = np.fft.ifft(y[k][i,:,j])
#*n/4
    out=f.from_list(y)                             # assemble tt from updated cores 
    out=out.round(eps,rma)
    return out

def bath_parameters():
    # Harmonic bath parameters for rhodopsin
    global wj,cj,qmodes,zem,cmodes
    global xc,pc,SACT,xxi,xxio,om,m
    cmodes=23
    if qmodes > 0:
        cmodes=0
    wj=[792.8,842.8,866.2,882.4,970.3,976.0,997.0,1017.1,1089.6,1189.0,\
            1214.7,1238.1,1267.9,1317.0,1359.0,1389.0,1428.4,1434.9,1451.8,\
            1572.8,1612.1,1629.2,1659.1]
    cj=[0.175,0.2,0.175,0.225,0.55,0.3,0.33,0.45,0.125,0.175,0.44,0.5,0.475,\
            0.238,0.25,0.25,0.25,0.225,0.225,0.25,0.225,0.125,0.225]

    xc=np.zeros(cmodes,dtype=float)  # classical bath coords
    pc=np.zeros(cmodes,dtype=float)  # classical bath momenta
    SACT=0                           # classical action
    xxi=1                            # survival amplitude for classical harmonic bath
    xxio=xxi                         # copy
    for j in range(23):
        wj[j]=wj[j]*4.5563353E-6
        cj[j]=cj[j]*wj[j]
    for j in range(qmodes):          # Parameters for bath harmonic modes
        om[j+2]=wj[j]
        m[j+2]=1.0/wj[j]
    zem=0                            # quantum zero point energy
    for j in range(qmodes):
        zem=zem+0.5*wj[j]

    return ()

def bath_propagation(tt_ps):
    # classical bath
    global wj,cj,zem,cmodes,tau
    global xc,pc,ddx,Vb1,Vb2,xxi,xxio,EYE,SACT
    Vb1=0.0
    Vb2=0.0
    xxio=xxi
    xxi=1
    aj = np.real(tt.dot(tt_ps,tt_ps))*ddx
    # Velocity Verlet propagation of bath coords and momenta, action and classical bath survival amplitude
    for j in range(cmodes):
        Vb1=Vb1+0.5*wj[j]*(xc[j]**2+pc[j]**2)
        Vb2=Vb2+0.5*wj[j]*(xc[j]**2+pc[j]**2)+cj[j]*xc[j]
        force=-xc[j]*wj[j]-aj*cj[j]
        xc[j]=xc[j]+wj[j]*pc[j]*tau+0.5*tau**2*force*wj[j]
        force2=-xc[j]*wj[j]-aj*cj[j]
        pc[j]=pc[j]+tau*0.5*(force+force2)
        xxi=xxi*exp(-(pc[j]**2+xc[j]**2+2.0*EYE*pc[j]*xc[j])/4.0)
        SACT=SACT+tau*pc[j]**2*wj[j]
    xxi=xxi*exp(EYE*SACT)
    return ()

def psio(r):
    # initial Gaussian state
    global dim,ro,po,EYE,sig,eps
    out=0
    for j in range(dim):
        out = out + ((r[:,j]-ro[j])/sig[j])**2
    out=out+np.sum(EYE*(r-ro)*po,axis=1)
    out=np.exp(-out)*(2.0/np.pi)**(0.25*dim)
    for j in range(dim):
        out=out/np.sqrt(sig[j])
    return out

def bra(z):
    global nstates,jind
    nevals, dim = z.shape
    M=np.zeros((nstates,nstates))
    out = np.zeros((nevals,))
    for ii in range(nevals):
        M[0,0] = z[ii,0]
        M[0,1] = z[ii,1]
        M[1,0] = z[ii,2]
        M[1,1] = z[ii,3]
        eval, evec = np.linalg.eig(M)
        if jind == 0:
            out[ii] =  evec[0,0]
        if jind == 1:
            out[ii] =  evec[0,1]
        if jind == 2:
            out[ii] =  evec[1,0]
        if jind == 3:
            out[ii] = evec[1,1]
    return out

def Up(p):
    # KE part of Trotter expansion
    global EYE,tau,m,dim
    out=0
    for j in range(dim):
        out = out + p[:,j]**2/(2*m[j])
    out = np.exp(-EYE*out*tau) 
    return out

def PEprop(tt_x):
    # PE propagator 
    global dim,nx,nstates,EYE,tau,eps,ttzeros,rma

    tt_v1 = tt.multifuncrs(tt_x, v1, eps,verb=0,rmax=rma)
    tt_v2 = tt.multifuncrs(tt_x, v2, eps,verb=0,rmax=rma)
    tt_vc = tt.multifuncrs(tt_x, vc, eps,verb=0,rmax=rma)

    onem=[] # identity matrix for exp Taylor expansion
    ttVm=[] # PE matrix valued tt
    ttones=tt.ones(nx[0],dim)
    ttzeros=ttones*0
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                onem.append(ttones)
                if i == 1:                  # switched order of states
                        ttVm.append(tt_v1)
                else:
                    ttVm.append(tt_v2)
            else:
                onem.append(ttzeros)
                ttVm.append(tt_vc)
    ttVm=np.reshape(ttVm,[nstates,nstates])
    onem=np.reshape(onem,[nstates,nstates])
    out=expA(ttVm*(-EYE*tau/2),onem,eps)
    ttVm=np.reshape(ttVm,nstates*nstates)
    return (out,ttVm)

def v1(r): 
    # PES 1
    global wj,cj,qmodes,zem,Vb1
    V1=3.6/2.*(1.-np.cos(r[:,0]))-eshift
    out=V1+0.19/2.*r[:,1]**2
    out=out/27.2
    # add quantum bath when qmodes > 0
    for j in range(qmodes):
        out = out + 0.5*wj[j]*r[:,j+2]**2
    # add classical bath when cmodes > 0
    out = out + Vb1
    # substract quantum zero-point energy
    out=out-zem
    return out

def v2(r):
    # PES 2
    global wj,cj,qmodes,zem,Vb2
    V2= 2.48-1.09/2.0*(1.0-np.cos(r[:,0]))-eshift
    out = V2+0.19/2.*r[:,1]**2+0.1*r[:,1]
    out=out/27.2
    # add quantum bath
    for j in range(qmodes):
        out = out + 0.5*wj[j]*r[:,j+2]**2+cj[j]*r[:,j+2]
    # add classical bath
    out = out + Vb2
    # substract quantum zero-point energy
    out=out-zem
    return out

def vc(r):
    # Coupling of PESs
    out = 0.19*r[:,1]
    out=out/27.2
    return out

def htrans(r):
    # Heaviside function for trans population
    global eps
    pi2=0.5*np.pi
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(np.abs(r[ii,0]) < pi2):
            out[ii]=1
    return out

def hcis(r):
    # Heaviside function for cis population
    global eps
    pi2=0.5*np.pi
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(np.abs(r[ii,0]) > pi2):
            out[ii]=1
    return out

def pop(tt_psi1,tt_heaviside):
    # Population defined by Heaviside function
    global ddx,eps,rma
    temp=tt_heaviside*tt_psi1*ddx
#    temp=temp.round(eps,rma)
    out = tt.dot(tt_psi1,temp)
    return np.real(out)

def genlist(e,i,dim,xone,oney):
    # generator of tt list of coordinates
    if i > 1:
        w = oney
        for j in range(i-1):
            w = tt.kron(e,w)
        for j in range(dim-1-i):
            w = tt.kron(w,e)
    else:
        if i == 0:
            w = xone
        else:
            w = oney
        for j in range(dim-2):
            w = tt.kron(w,e)
    return w

# main program
if __name__ == "__main__":
    global n,nx,EYE,tau,eps,rma,dim,ro,po,rd,wfflag,ddx,jind,aflag
    global dx,dp,m,om,nsc,Pxy,gam,d,xxio,qmodes,cmodes,eshift
    parameters()                      # read parameters
    bath_parameters()                 # initialize bath 
    # build grids
    xv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dx[0]))  
    yv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dx[1])) 
    pxv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dp[0])) 
    pyv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dp[1]))
    xs=np.fft.fftshift(xv)
    ys=np.fft.fftshift(yv)
    x,y=np.meshgrid(xs, ys, sparse=False, indexing='ij')
    px,py=np.meshgrid(pxv, pyv, sparse=False, indexing='ij')

    # grids for building tensor trains
    xone =  np.zeros(nx[0]*nx[1],dtype=float)
    oney =  np.zeros(nx[0]*nx[1],dtype=float)
    pxone = np.zeros(nx[0]*nx[1],dtype=float)
    onepy = np.zeros(nx[0]*nx[1],dtype=float)
    for i in range(nx[0]):
        for j in range(nx[1]):
            ind=i+nx[0]*j
            xone[ind]=xs[j]
            oney[ind]=ys[i]
    for i in range(nx[0]):
        for j in range(nx[1]):
            ind=i+nx[0]*j
            pxone[ind]=pxv[j]
            onepy[ind]=pyv[i]
    # tt ones and zeros
    rones = tt.ones(nx[0],1)

    # coord tt_x list
    if True:
        ttxone=tt.tensor(np.reshape(xone,[nx[0],nx[1]]))
        ttoney=tt.tensor(np.reshape(oney,[nx[0],nx[1]]))
        tt_x = [genlist(rones,i,dim,ttxone,ttoney) for i in range(dim)]

    # momenta tt_p list
    if True:
        ttpxone=tt.tensor(np.reshape(pxone,[nx[0],nx[1]]))
        ttonepy=tt.tensor(np.reshape(onepy,[nx[0],nx[1]]))
        tt_p = [genlist(rones,i,dim,ttpxone,ttonepy) for i in range(dim)]

    # initial tt_psi state
    if True:
        tt_psi1=tt.multifuncrs2(tt_x, psio,verb=0)
    if True:
        tt_psi2=tt_psi1*0
        tt_psi=[]
        tt_psi.append(tt_psi1)       # populate state 1
        for i in range(1,nstates):
            tt_psi.append(tt_psi2)   # no initial population in other states
        overlap = np.real(tt.dot(tt_psi1,tt_psi1))*ddx
        print("initial overlap=",overlap)

    if True:
        # trans and cis Heaviside functions
        tt_trans=tt.multifuncrs(tt_x, htrans,verb=0)
        tt_cis=tt.multifuncrs(tt_x, hcis,verb=0)
    if True:
        # initial cis and trans populations on excited state
        print("initial trans, cis pops=", pop(tt_psi1,tt_trans), pop(tt_psi1,tt_cis))

    if True:
        # KE part of Trotter expansion
        tt_Pxy =tt.multifuncrs(tt_p, Up,verb=0)
    if True:
        # PE propator 
        UV,tt_d=PEprop(tt_x)
        if aflag == 1:
            # adiabatic eigenvectors
            tt_bra=[]
            for jind in range(4):
                tt_bra.append(tt.multifuncrs(tt_d, bra, eps,verb=0,rmax=3))
            # initialize survival amplitude rt=<psi0|psit>
        rt = np.zeros((10*nsc),dtype=complex)  

        # initialize populations trans (ppt0,ppt1) and cis (ppc0,ppc1)
        ptransS0 = np.zeros((nsc))
        ptransS1 = np.zeros((nsc))  
        pcisS0 = np.zeros((nsc))  
        pcisS1 = np.zeros((nsc))  

    # array of times for visualization of survival amplitude
        au2ps=0.00002418884254 # conversion factor from au to ps
        t=np.linspace(0,10*nsc,10*nsc)*tau*au2ps
    # save a copy of initial state
        tt_psi0=tt_psi
        plt.figure(figsize=(7,5))
        nsl=np.int(nx[0]/2) # index of slices for visualization of multidimensional wavepackets

    if True:
    # Propagation loop
        for js in range(nsc):   
            if True:
                # compute survival amplitude rt=<psi0|psit>
                rr=0
                for i in range(nstates):
                    rr=rr+tt.dot(tt_psi0[i],tt_psi[i])*ddx
                rt[js]=rr
                # with classical modes, Hamiltonian is time-dependent (time-dependent bath)
                if cmodes > 0:
                    # update propagator for time evolved bath
                    UV,tt_d=PEprop(tt_x)
                    # evolve bath
                    bath_propagation(tt_psi[0])
                    # update adiabatic eigenvectors
#                    for jind in range(4):
#                        tt_bra.append(tt.multifuncrs(tt_d, bra, eps,verb=0,rmax=5))
            if True:
                # adiabatic populations
                if aflag == 1:
                    # adiabatic states
                    tta =[]
                    temp=tt_bra[0]*tt_psi[0]+tt_bra[1]*tt_psi[1]
                    tta.append(temp.round(eps,rma))
                    temp=tt_bra[2]*tt_psi[0]+tt_bra[3]*tt_psi[1]
                    tta.append(temp.round(eps,rma))
                    # adiabatic populations
                    ptransS1[js] = pop(tta[0],tt_trans)
                    ptransS0[js] = pop(tta[1],tt_trans)
                    pcisS1[js] = pop(tta[0],tt_cis)
                    pcisS0[js] = pop(tta[1],tt_cis)
                    
                # diabatic populations
                if aflag == 0:
                    ptransS1[js] = pop(tt_psi[0],tt_trans)
                    ptransS0[js] = pop(tt_psi[1],tt_trans)
                    pcisS1[js] = pop(tt_psi[0],tt_cis)
                    pcisS0[js] = pop(tt_psi[1],tt_cis)

                ptot=ptransS1[js]+ptransS0[js]+pcisS1[js]+pcisS0[js]
                ptransS1[js]=ptransS1[js]/ptot
                ptransS0[js]=ptransS0[js]/ptot
                pcisS1[js]=pcisS1[js]/ptot
                pcisS0[js]=pcisS0[js]/ptot
                print("ptot=",ptot)
                
                rt[js]=rt[js]*xxio
                tt_psi=tt_soft(tt_psi,UV,tt_Pxy)

            # plot wavepacket components
            if True:
                if wfflag == 1:
                    ttpsi1=tt_psi[0].full()
                    ttpsi2=tt_psi[1].full()
                    ttpsi1=np.reshape(ttpsi1,[nx[0]]*dim)
                    ttpsi2=np.reshape(ttpsi2,[nx[0]]*dim)
                    ax= plt.subplot(3,2,2, projection='3d')
                    if  qmodes == 0:
                        ax.plot_surface(x, y, np.abs(ttpsi2[:,:]), cmap=cm.coolwarm, antialiased=False)
                    if  qmodes == 1:
                        ax.plot_surface(x, y, np.abs(ttpsi2[:,:,nsl]), cmap=cm.coolwarm, antialiased=False)
                    ax.set_zlim3d(0,1)
                    ax= plt.subplot(3,2,1, projection='3d')
                    if  qmodes == 0:
                        ax.plot_surface(x, y, np.abs(ttpsi1[:,:]), cmap=cm.coolwarm, antialiased=False)
                    if  qmodes == 1:
                        ax.plot_surface(x, y, np.abs(ttpsi1[:,:,nsl]), cmap=cm.coolwarm, antialiased=False)
                    ax.set_zlim3d(0,1)
#                    plt.pause(.02)
#                    if js < nsc-1:
#                        plt.clf()
            if True:
                # plot survival amplitude
                if wfflag == 1:
                    ax= plt.subplot(3,2,3)
                else:
                    ax= plt.subplot(3,1,1)
                ax.plot(t[:js],np.real(rt[:js]),label='real(rt)')
                ax.plot(t[:js],np.imag(rt[:js]),label='imag(rt)')
                ax.plot(t[:js],np.abs(rt[:js]),label='abs(rt)')
                plt.legend()
                ax.set_ylim(-1,1)
                ax.grid()
                if wfflag == 1:
                    ax= plt.subplot(3,2,4)
                else:
                    ax= plt.subplot(3,1,2)
                ax.plot(t[:js],np.real(ptransS1[:js]),label='Pop_trans[S1]')
                ax.plot(t[:js],np.real(ptransS0[:js]),label='Pop_trans[S0]')
                ax.plot(t[:js],np.real(pcisS1[:js]),label='Pop_cis[S1]')
                ax.plot(t[:js],np.real(pcisS0[:js]),label='Pop_cis[S0]')
                ax.grid()
                plt.legend()
                plt.pause(.02)
                if js < nsc-1:
                    plt.clf()

        if True:
            # compute spectrum as the FT of rt
            dw=2.0*np.pi/(tau*10*nsc)
            w=np.fft.fftshift(np.fft.fftfreq(10*nsc,1.0/(10*nsc*dw)))
#            rt= rt*np.exp(-1j*t*tau*eshift/27.2/au2ps/tau)
            rw=np.fft.fftshift(np.fft.ifft(rt*np.exp(-tau*t*gam))) 
#            rw=np.fft.fftshift(np.fft.ifft(rt*np.exp(-tau*t*(gam)))) 
            if wfflag == 1:
                ax= plt.subplot(3,2,5)
            else:
                ax= plt.subplot(3,1,3)
            ax.plot(w*27.2+eshift,np.real(rw),label='Absorption Spectrum')
            plt.legend()
            ax.set_xlim(2,3)
            ax.grid()
            plt.pause(15.)
            
        if True:
            # save survival amplitude and populations
            savetxt('times.npy', t[:nsc])
            if qmodes >0:  # for full quantum results
                savetxt('qrt_real.npy', np.real(rt[:nsc]))
                savetxt('qrt_imag.npy', np.imag(rt[:nsc]))
                savetxt('qppt0.npy', np.real(ptransS0[:nsc]))
                savetxt('qppt1.npy', np.real(ptransS1[:nsc]))
                savetxt('qppc1.npy', np.real(pcisS1[:nsc]))
                savetxt('qppc0.npy', np.real(pcisS0[:nsc]))
            if cmodes >0:  # for TDSCF results
                savetxt('crt_real.npy', np.real(rt[:nsc]))
                savetxt('crt_imag.npy', np.imag(rt[:nsc]))
                savetxt('cppt0.npy', np.real(ptransS0[:nsc]))
                savetxt('cppt1.npy', np.real(ptransS1[:nsc]))
                savetxt('cppc1.npy', np.real(pcisS1[:nsc]))
                savetxt('cppc0.npy', np.real(pcisS0[:nsc]))
                
