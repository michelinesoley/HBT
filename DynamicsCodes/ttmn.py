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
import random
import pylab
import math
import tt
import time
from numpy import linalg as LA

from HBT_potential import init_pot,reactive_potential,bath_displacement,bath_coupling

def parameters():
    global nstates,n,nx,EYE,tau,eps,rma,dim,rd,ro,po
    global dx,dp,nsc,gam,d,ddx
    global sig,m,om,eshift
    global wj,cj,qmodes,Vb1,Vb2,aflag
    global specshift
    global minpos,L
    global displacements,coupling

    qmodes = 3 # 67              # qmodes = 0 for TDSCF simulation
    nsc = 800             # number of propagation steps
    tau = 12.5            # propagation time step
    gam = 1e-7            # phenomeological dumping factor for simulating homogeneous broadening
    eps = 1e-14            # tt approx error
    rma = 10                # max tt rank
    eshift = 0          # energy shift for PESs
    specshift = 0 # spectrum energy shift in au
    dim = 2+qmodes         # number of coords
    nstates=2              # number of surfaces
    d = 5
    n = 2**d               # number or grid points
    Lx = 200 # box size x
    Ly = 500 # box size y
    L = np.ones((dim))
    L[0] = Lx
    L[1] = Ly
    for ii in range(qmodes):
      L[ii+2]=20.*np.sqrt(1./np.sqrt(coupling[ii]))
    ro=np.zeros(dim,dtype=float) # np.zeros(dim,dtype=float)       # initial wavepacket position
    ro[0]=-37.5+5 # Shifted ground state ground state wavefunction
    ro[1]=103.515625+20
    for ii in range(qmodes):
      ro[ii+2]=displacements[ii]
    po=np.ones(dim,dtype=float)*0.     # initial wavepacket momentum
    nx=np.zeros(dim,dtype=int)         # number of grid points per dimension
    dx=np.zeros(dim,dtype=float)       # grid point spacing
    dp=np.zeros(dim,dtype=float)       # momentum grid point spacing
    minpos=np.zeros(dim,dtype=float) # minimum position
    minpos[0]=-L[0]/2.
    minpos[1]=-L[1]/2.
    for i in range(qmodes):
      minpos[i+2]= displacements[i]-L[i+2]/2
    for i in range(dim):
        nx[i] = n                      # number of grid points
        dx[i] = L[i]/(nx[i]-1)             # coord grid spacing
        dp[i] = 2.0*np.pi/L[i]         # momenta grid spacing
    ddx=1.0
    for i in range(dim):
        ddx=ddx*dx[i]
    EYE = complex(0,1)                 # imaginary number
    m = np.ones((dim))                 # masses
    om = np.ones((dim))                # frequencies
    sig = np.ones((dim))*np.sqrt(2.0)  # Gaussian widths

    # Parameters for the first 2 modes (large amplitude modes of reaction surface, theta and x_str)
    sig[0] = 22.473218269408434 # HBT parameters
    sig[1] = 53.13162239494416

    # Parameters for bath modes
    for ii in range(qmodes):
      om[ii+2]=np.sqrt(coupling[ii])
      sig[ii+2]=np.sqrt(1./om[ii])*np.sqrt(2.0)
    expeczpe=0.
    for ii in range(qmodes):
      expeczpe=expeczpe+om[ii+2]
    expeczpe=expeczpe/2.
    expeczpe=expeczpe+0.00233425640548065 # Ground state large-amplitude mode ZPE
    print("Expected ZPE=",expeczpe)
    
    return()

def mv22(emat,psi):
    # nstates x nstates matrix times vector valued tensor trains
    global eps,rma,ttzeros,nstates
    out=[]
    for j in range(nstates):
        out.append(ttzeros)
        for k in range(nstates):
            out[j] = out[j] + (emat[j,k]*psi[k]).round(eps,rma)
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
    # SOFT propagation
    global nstates,eps,rma
    out=mv22(emat,fxy)
    for j in range(nstates):
        fp=mfft(out[j],1)*Pxy
        if False: # WARNING: THIS LINE CREATES A WARNING ABOUT CASTING COMPLEX VALUES TO REAL
            fp=fp.round(eps,rma)
        out[j]=mfft(fp,-1)
    out=mv22(emat,out)
    return out

def tt_potenergy(fxy,emat):
    # Potential energy operator application
    global nstates,eps,rma
    potpart=mv22(emat,fxy)
    out=potpart
    return out

def tt_kinenergy(fxy,Pxy):
    # Kinetic energy operator application
    global nstates,eps,rma
    out=[]
    for j in range(nstates):
        out.append(ttzeros)
        out[j] = out[j] + fxy[j]
        out[j] = out[j].round(eps,rma)
    for j in range(nstates):
        fp=mfft(out[j],1)*Pxy
        fp=fp.round(eps,rma)
        out[j]=mfft(fp,-1)
    return out

def Kin(p):
    # Kinetic energy
    global m,dim
    out=0
    for j in range(dim):
        out = out + p[:,j]**2/(2*m[j])
    return out

def mfft(f,ind):
    # Multidimensional FFT of function f in TT format
    # ind=1 for FFT, otherwise IFFT
    global eps, rma
    y=f.to_list(f)                                 # Get cores - translate TT to list
    for k in range(len(y)):                        # Core index - dimension
        for i in range(y[k].shape[0]):             # Left auxiliary index of core k
            for j in range(y[k].shape[2]):         # Right auxiliary index of core k
                if ind == 1:
                    y[k][i,:,j] = np.fft.fft(y[k][i,:,j]) # Perform fft in one direction # WARNING: THIS LINE CREATES A WARNING ABOUT CASTING COMPLEX VALUES TO REAL ON THE FIRST STEP (I.E., WHEN THERE IS NO POPULATION IN THE EXCITED STATE)
#*4/n
                else:
                    y[k][i,:,j] = np.fft.ifft(y[k][i,:,j]) # Perform fft in one direction
#*n/4
    out=f.from_list(y)                             # Assemble tt from updated cores                                             # - translate list to TT
    out=out.round(eps,rma)
    return out

def psio(r):
    # Initial Gaussian state
    global dim,ro,po,EYE,sig,eps
    out=0
    for j in range(dim):
        out = out + ((r[:,j]-ro[j])/sig[j])**2
    out=out+np.sum(EYE*(r-ro)*po,axis=1)
    out=np.exp(-out)*(2.0/np.pi)**(0.25*dim)
    for j in range(dim):
        out=out/np.sqrt(sig[j])
    nevals, size = r.shape
    for ii in range(nevals):
      out[ii]=out[ii]+1e-5*random.uniform(-1,1) # Noise to avoid difficulties in cross approximation
    return out

def Up(p):
    # Kinetic energy part of Trotter expansion
    global EYE,tau,m,dim
    global direction
    out=0
    for j in range(dim):
        out = out + p[:,j]**2/(2*m[j])
    out = np.exp(-direction*EYE*out*tau)
    return out

def PEprop(tt_x):
    # Potential energy part of Trotter expansion
    global dim,nx,nstates,EYE,tau,eps,ttzeros,rma
    global direction
    global tt_v1
    tt_v2 = tt_v1
    if True:
      tt_vc = tt.multifuncrs(tt_x, vc, eps,verb=0,rmax=rma)

    onem=[] # Identity matrix for Taylor expansion
    ttVm=[] # Potential energy matrix-valued TT
    ttones=tt.ones(nx[0],dim)
    ttzeros=ttones*0
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                onem.append(ttones)
                if i == 1:                  # Switched order of states
                        ttVm.append(tt_v1)
                else:
                    ttVm.append(tt_v2)
            else:
                onem.append(ttzeros)
                ttVm.append(tt_vc)
    ttVm=np.reshape(ttVm,[nstates,nstates])
    onem=np.reshape(onem,[nstates,nstates])
    out=expA(ttVm*(-direction*EYE*tau/2),onem,eps)
    tt_d=np.reshape(ttVm,nstates*nstates)
    return (out,ttVm,tt_d)

def vc(r):
    # Coupling of PESs
    out = 1.e-18*r[:,1] # Nonzero to avoid difficulties in the cross approximation
    return out
  
def ttground(rr):
  # Generate a tensor train for the ground state
  global displacements,coupling
  global xminorig,xmaxorig,yminorig,ymaxorig
  nevals, dim = rr.shape # Calculate number of evaluations and dimensionality
  out = np.zeros((nevals,)) # Initialize array
  for ii in range(nevals):
    roundx=rr[ii,0]
    roundy=rr[ii,1]
    if roundx < xminorig:
      roundx=xminorig
    elif roundx > xmaxorig:
      roundx=xmaxorig
    if roundy < yminorig:
      roundy=yminorig
    elif roundy > ymaxorig:
      roundy=ymaxorig
    displacements=bath_displacement(roundx,roundy,'gs')
    coupling=bath_coupling(0.,0.,'gs')
    out[ii]=reactive_potential(rr[ii,0],rr[ii,1],pot_surf='gs')
    for jj in range(qmodes):
      out[ii]=out[ii]+0.5*coupling[jj]*(rr[ii,jj+2]-displacements[jj])**2
  return out

def ttexcited(rr):
  # Generate a tensor train for the excited state
  global displacements,coupling
  global xminorig,xmaxorig,yminorig,ymaxorig
  nevals, dim = rr.shape # Calculate number of evaluations and dimensionality
  out = np.zeros((nevals,)) # Initialize array
  for ii in range(nevals):
    roundx=rr[ii,0]
    roundy=rr[ii,1]
    if roundx < xminorig:
      roundx=xminorig
    elif roundx > xmaxorig:
      roundx=xmaxorig
    if roundy < yminorig:
      roundy=yminorig
    elif roundy > ymaxorig:
      roundy=ymaxorig
    displacements=bath_displacement(roundx,roundy,'es')
    coupling=bath_coupling(0.,0.,'es')
    out[ii]=reactive_potential(rr[ii,0],rr[ii,1],pot_surf='es')
    for jj in range(qmodes):
      out[ii]=out[ii]+0.5*coupling[jj]*(rr[ii,jj+2]-displacements[jj])**2
  return out

def htrans(r):
    # Heaviside function for TS+enol population
    global eps
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(r[ii,0] <= 0):
            out[ii]=1
    return out

def hcis(r):
    # Heaviside function for keto population
    global eps
    pi2=0.5*np.pi
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(r[ii,0] > 0):
            out[ii]=1
    return out

def pop(tt_psi1,tt_heaviside):
    # Population defined by Heaviside function
    global ddx,eps,rma
    temp=tt_heaviside*tt_psi1*ddx
    out = tt.dot(tt_psi1,temp)
    return np.real(out)

def genlist(e,i,dim,xone,oney,onez):
    # generator of tt list of coordinates
    if i > 1:
        w = onez[i-2]
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
    
def tt_normalize(psi,normcttmat):
    # normalize vector-valued tensor trains
    global eps,rma,ttzeros,nstates
    out=[]
    for j in range(nstates):
        out.append(ttzeros)
        for k in range(nstates):
            out[j] = out[j] + normcttmat[j,k]*psi[k]
        out[j] = out[j].round(eps,rma)
    return out

def savewavefunction(tt_tensor,filename):
    # save wavefunction
    global nstates
    for ii in range(nstates):
      tttensor=tt_tensor[ii].to_list(tt_tensor[ii])
      tempfilename=filename+'state%i' % ii
      np.save(tempfilename,tttensor)
    return

# main program
if __name__ == "__main__":
    global n,nx,EYE,tau,eps,rma,dim,ro,po,rd,ddx
    global dx,dp,m,om,nsc,Pxy,gam,d,qmodes,eshift
    global minpos,L
    global displacements,coupling
    global xminorig,xmaxorig,yminorig,ymaxorig
    global xminhbt,xmaxhbt,yminhbt,ymaxhbt
    
    # Define maximum and minimum values of x and y axes in ab initio file
    xminorig=-50. # Minimum value of x axis
    xmaxorig=60. # Maximum value of x axis
    yminorig=-140. # Minimum value of y axis
    ymaxorig=240. # Maximum value of y axis
    
    # Define grid for reading HBT potential and save potential
    nhbt=2**5
    xminhbt=-100.
    xmaxhbt=100.
    dxhbt=(xmaxhbt-xminhbt)/(nhbt-1)
    yminhbt=-250.
    ymaxhbt=250.
    dyhbt=(ymaxhbt-yminhbt)/(nhbt-1)
    xshbt=np.arange(xminhbt,xmaxhbt+dxhbt/2,dxhbt)
    yshbt=np.arange(yminhbt,ymaxhbt+dyhbt/2,dyhbt)

    init_pot(xshbt,yshbt,pot_surf='gs',pot_type='diag_fix') # Initialize ground state
    init_pot(xshbt,yshbt,pot_surf='es',pot_type='diag_fix') # Initialize excited state

    displacements=bath_displacement(0.,0.,'gs')
    coupling=bath_coupling(0.,0.,'gs')
    
    parameters()                      # read parameters
    
    # build grids
    pxv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dp[0])) 
    pyv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dp[1]))
    pzv=[np.fft.fftfreq(nx[ii+2],1.0/(nx[ii+2]*dp[ii+2])) for ii in range(qmodes)]
    xs=np.arange(minpos[0],minpos[0]+L[0]+dx[0],dx[0])
    ys=np.arange(minpos[1],minpos[1]+L[1]+dx[0],dx[1])
    zs=[np.arange(minpos[ii+2],minpos[ii+2]+L[ii+2]+dx[ii+2],dx[ii+2]) for ii in range(qmodes)]
    x,y=np.meshgrid(xs, ys, sparse=False, indexing='ij')
    x=np.zeros((n,n))
    y=np.zeros((n,n))
    for ii in range(n):
      x[ii,:]=xs[ii]
      y[:,ii]=ys[ii]

    # grids for building tensor trains
    xone =  np.zeros(nx[0]*nx[1],dtype=float)
    oney =  np.zeros(nx[0]*nx[1],dtype=float)
    onez = [np.zeros(nx[0]*nx[1],dtype=float) for ii in range(qmodes)] # assumes all grid dimensions equal
    pxone = np.zeros(nx[0]*nx[1],dtype=float)
    onepy = np.zeros(nx[0]*nx[1],dtype=float)
    onepz = [np.zeros(nx[0]*nx[1],dtype=float) for ii in range(qmodes)] # assumes all grid dimensions equal
    for i in range(nx[0]):
        for j in range(nx[1]):
            ind=i+nx[0]*j
            xone[ind]=xs[j]
            oney[ind]=ys[i]
            for ii in range(qmodes):
              onez[ii][ind]=zs[ii][i]
    for i in range(nx[0]):
        for j in range(nx[1]):
            ind=i+nx[0]*j
            pxone[ind]=pxv[j]
            onepy[ind]=pyv[i]
            for ii in range(qmodes):
              onepz[ii][ind]=pzv[ii][i]
    # tt ones and zeros
    rones = tt.ones(nx[0],1) # Assumes number of grid points equal in all dimensions

    # coord tt_x list
    if True:
        ttxone=tt.tensor(np.reshape(xone,[nx[0],nx[1]]))
        ttoney=tt.tensor(np.reshape(oney,[nx[0],nx[1]]))
        ttonez=[tt.tensor(np.reshape(onez[ii],[nx[0],nx[1]])) for ii in range(qmodes)] # Assumes number of grid points equal in all dimensions
        tt_x = [genlist(rones,i,dim,ttxone,ttoney,ttonez) for i in range(dim)]

    # momenta tt_p list
    if True:
        ttpxone=tt.tensor(np.reshape(pxone,[nx[0],nx[1]]))
        ttonepy=tt.tensor(np.reshape(onepy,[nx[0],nx[1]]))
        ttonepz=[tt.tensor(np.reshape(onepz[ii],[nx[0],nx[1]])) for ii in range(qmodes)] # Assumes number of grid points equal in all dimensions
        tt_p = [genlist(rones,i,dim,ttpxone,ttonepy,ttonepz) for i in range(dim)]

    # initial tt_psi state
    if True:
        tt_psi1=tt.multifuncrs2(tt_x, psio,verb=0)
        tt_psi1=tt_psi1.round(eps,rma)
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
        print("initial keto, TS+enol pops=", pop(tt_psi1,tt_trans), pop(tt_psi1,tt_cis))
    # Set up potential definitions
    
    if True:
      print("Beginning imaginary time propagation...")

      # Imaginary time propagation
      direction=-1j # 1 # real-time propagation = 1, imaginary-time propagation = -1j
      tt_v1 = tt.multifuncrs(tt_x, ttground, eps,verb=0,rmax=rma) # Define initial states

      if True:
          # KE part of Trotter expansion
          tt_Pxy =tt.multifuncrs(tt_p, Up,verb=0)
      if True:
          # PE propator
          UV,ttVm,tt_d=PEprop(tt_x)
          x0 = np.zeros((nsc),dtype=float)
          x1 = np.zeros((nsc),dtype=float)
          xn = np.zeros((qmodes,nsc),dtype=float)

          # initialize populations trans (ppt0,ppt1) and cis (ppc0,ppc1)
          ptransS0 = np.zeros((nsc))
          ptransS1 = np.zeros((nsc))
          pcisS0 = np.zeros((nsc))
          pcisS1 = np.zeros((nsc))

      # array of times for visualization of survival amplitude
          au2ps=0.00002418884254 # conversion factor from au to ps
          au2fs=0.02418884254
          t=np.linspace(0,10*nsc,10*nsc)*tau*au2fs
      # save a copy of initial state
          tt_psi0=tt_psi
          plt.figure(figsize=(7,5))
          nsl=np.int(nx[0]/2) # index of slices for visualization of multidimensional wavepackets, assumes all grid dimensions equal
          
      if True:
      # Propagation loop
          for js in range(nsc):
              print("Step No.: ",js)
              if True:
                  expecx0=0
                  expecx1=0
                  expecxn=np.zeros((qmodes))
                  for i in range(nstates):
                      ttxpsi=tt_x[0]*tt_psi[i]
                      ttypsi=tt_x[1]*tt_psi[i]
                      expecx0=expecx0+tt.dot(tt_psi[i],ttxpsi)*ddx
                      expecx1=expecx1+tt.dot(tt_psi[i],ttypsi)*ddx
                      for jj in range(qmodes):
                          expecxn[jj]=expecxn[jj]+np.real(tt.dot(tt_psi[i],tt_x[2+jj]*tt_psi[i])*ddx)
                  x0[js]=np.real(expecx0)
                  x1[js]=np.real(expecx1)
                  for jj in range(qmodes):
                    xn[jj,js]=np.real(expecxn[jj])
                    
              if True:
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
                
                  tt_psi=tt_soft(tt_psi,UV,tt_Pxy)
                
              if True:
                  # renormalize
                  overlap = 0
                  for ii in range(nstates):
                    overlap=overlap+np.real(tt.dot(tt_psi[ii],tt_psi[ii]))*ddx
                  print("overlap= ",overlap)
                  normctt=np.real(1./np.sqrt(overlap))
                  normcttmat=np.array([[normctt,0,],[0,normctt]])
                  tt_psi=tt_normalize(tt_psi,normcttmat)
              
              if True:
                  # print energy
                  potenergy=0
                  tt_psiV=tt_potenergy(tt_psi,ttVm)
                  for i in range(nstates):
                      potenergy=potenergy+tt.dot(tt_psi[i],tt_psiV[i])*ddx
                  
                  print('Potential Energy=',np.real(potenergy))
                  tt_Kin = tt.multifuncrs(tt_p, Kin, eps=eps,verb=0,rmax=rma)
                  tt_Kin = tt_Kin.round(eps,rma)
                  tt_psiT=tt_kinenergy(tt_psi,tt_Kin) # The problem lies here.
                  kinenergy=0
                  for i in range(nstates):
                      kinenergy=kinenergy+tt.dot(tt_psi[i],tt_psiT[i])*ddx
                  
                  print('Kinetic Energy=',np.real(kinenergy))
                  
                  print('Total Energy=',np.real(potenergy+kinenergy))
    
    print("Beginning real time propagation...")
      
    # Real time propagation

    direction=1 # 1 # real-time propagation = 1, imaginary-time propagation = -1j

    displacements=bath_displacement(0.,0.,'es')
    coupling=bath_coupling(0.,0.,'es')

    tt_v1 = tt.multifuncrs(tt_x, ttexcited, eps,verb=0,rmax=rma) # Define initial states

    if True:
        # KE part of Trotter expansion
        tt_Pxy =tt.multifuncrs(tt_p, Up,verb=0)
    if True:
        # PE propator
        UV,ttVm,tt_d=PEprop(tt_x)
        x0 = np.zeros((nsc),dtype=float)
        x1 = np.zeros((nsc),dtype=float)
        for jj in range(qmodes):
          xn[jj]=np.real(expecxn[jj])

        # initialize populations trans (ppt0,ppt1) and cis (ppc0,ppc1)
        ptransS0 = np.zeros((nsc))
        ptransS1 = np.zeros((nsc))
        pcisS0 = np.zeros((nsc))
        pcisS1 = np.zeros((nsc))
        
        # initialize norm and energy checks
        normcheck = np.zeros((nsc))
        energycheck = np.zeros((nsc))
        potcheck = np.zeros((nsc))
        kincheck = np.zeros((nsc))

    # array of times for visualization of survival amplitude
        au2ps=0.00002418884254 # conversion factor from au to ps
        au2fs=0.02418884254
        t=np.linspace(0,10*nsc,10*nsc)*tau*au2fs
    # save a copy of initial state
        tt_psi0=tt_psi
        plt.figure(figsize=(7,5))
        nsl=np.int(nx[0]/2) # index of slices for visualization of multidimensional wavepackets
        
    if True:
        savewavefunction(tt_psi,'realprop0')
        
    if True:
    # Propagation loop
        for js in range(nsc):
            print("Step No.: ",js)
            if True:
                expecx0=0
                expecx1=0
                expecxn=np.zeros((qmodes))
                for i in range(nstates):
                    ttxpsi=tt_x[0]*tt_psi[i]
                    ttypsi=tt_x[1]*tt_psi[i]
                    expecx0=expecx0+tt.dot(tt_psi[i],ttxpsi)*ddx
                    expecx1=expecx1+tt.dot(tt_psi[i],ttypsi)*ddx
                    for jj in range(qmodes):
                        expecxn[jj]=expecxn[jj]+np.real(tt.dot(tt_psi[i],tt_x[2+jj]*tt_psi[i])*ddx)
                x0[js]=np.real(expecx0)
                x1[js]=np.real(expecx1)
                for jj in range(qmodes):
                  xn[jj,js]=np.real(expecxn[jj])
                    
            if True:
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
                
                tt_psi=tt_soft(tt_psi,UV,tt_Pxy)
            
            if True:
                # renormalize
                overlap = 0
                for ii in range(nstates):
                  overlap=overlap+np.real(tt.dot(tt_psi[ii],tt_psi[ii]))*ddx
                print("overlap= ",overlap)
                normcheck[js]=overlap
                normctt=np.real(1./np.sqrt(overlap))
                normcttmat=np.array([[normctt,0,],[0,normctt]])
                tt_psi=tt_normalize(tt_psi,normcttmat)
                
            if True:
                # print energy
                potenergy=0
                tt_psiV=tt_potenergy(tt_psi,ttVm)
                for i in range(nstates):
                    potenergy=potenergy+tt.dot(tt_psi[i],tt_psiV[i])*ddx
                
                print('Potential Energy=',np.real(potenergy))
                tt_Kin = tt.multifuncrs(tt_p, Kin, eps=eps,verb=0,rmax=rma)
                tt_Kin = tt_Kin.round(eps,rma)
                tt_psiT=tt_kinenergy(tt_psi,tt_Kin) # The problem lies here.
                kinenergy=0
                for i in range(nstates):
                    kinenergy=kinenergy+tt.dot(tt_psi[i],tt_psiT[i])*ddx
                  
                print('Kinetic Energy=',np.real(kinenergy))
                  
                print('Total Energy=',np.real(potenergy+kinenergy))
                potcheck[js]=np.real(potenergy)
                kincheck[js]=np.real(kinenergy)
                energycheck[js]=np.real(potenergy+kinenergy)
                    
            if True:
                newfilename='realprop%i' % js
                savewavefunction(tt_psi,newfilename)
    
    # save time-stamped data files
        
    # save expectation values
    filename=open("expectationvalues.dat","w+")
    for ii in range(nsc):
      print(t[ii],"  ",np.real(x0[ii]),"  ",np.real(x1[ii]),"  ",*np.real(xn[:,ii]),file=filename)
    filename.close()

    # save populations
    filename=open("populations.dat","w+")
    for ii in range(nsc):
      print(t[ii],"  ",np.real(ptransS0[ii]),"  ",np.real(ptransS1[ii]),"  ",np.real(pcisS1[ii]),"  ",np.real(pcisS0[ii]),file=filename)
    filename.close()
    
    # save norm
    filename=open("norm.dat","w+")
    for ii in range(nsc):
      print(t[ii],"  ",np.real(normcheck[ii]),file=filename)
    filename.close()

    # save energy
    filename=open("energy.dat","w+")
    for ii in range(nsc):
      print(t[ii],"  ",np.real(kincheck[ii]),"  ",np.real(potcheck[ii]),"  ",np.real(energycheck[ii]),file=filename)
    filename.close()
