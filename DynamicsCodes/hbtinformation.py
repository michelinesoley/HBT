import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tt
from HBT_potential import init_pot,reactive_potential,bath_displacement,bath_coupling

# Program to read a file with potential parameters and save them in
# tensor train format

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

def ttground(rr): # Generate a tensor train for the ground state
  nevals, dim = rr.shape # Calculate number of evaluations and dimensionality
  out = np.zeros((nevals,)) # Initialize array
  for ii in range(nevals):
    out[ii]=reactive_potential(rr[ii,0],rr[ii,1],pot_surf='gs')
  return out
  
def ttexcited(rr): # Generate a tensor train for the excited state
  nevals, dim = rr.shape # Calculate number of evaluations and dimensionality
  out = np.zeros((nevals,)) # Initialize array
  for ii in range(nevals):
    out[ii]=reactive_potential(rr[ii,0],rr[ii,1],pot_surf='es')
  return out

def ttdisplacements(rr): # Generate a tensor train for the displacements
  global specmode
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
    out[ii]=displacements[specmode]
  return out

def ttcouplingsfull(rr): # Generate a tensor train for the couplings
  global specmode0,specmode1
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
    couplings=bath_coupling(roundx,roundy,'es')
    out[ii]=couplings[specmode0,specmode1]
  return out
  
def psio(r): # Generate a tensor train for the initial Gaussian state
  global dim,ro,po,EYE,sig
  out=0
  for j in range(dim):
    out = out + ((r[:,j]-ro[j])/sig[j])**2
  out=out+np.sum(EYE*(r-ro)*po,axis=1)
  out=np.exp(-out)*(2.0/np.pi)**(0.25*dim)
  for j in range(dim):
    out=out/np.sqrt(sig[j])
  return out
      

global ttjustone,dim,qmodes,ttxone,ttoney
global minpos,dx,potentialground,potentialexcited
global xminorig,xmaxorig,yminorig,ymaxorig
global dim,ro,po,EYE,sig
global specmode
global specmode0,specmode1
global displacements,coupling

# Define math parameters
EYE = 1j

# Define number of grid divisions
qmodes = 0 # Set to 0 to print all images
dim = 2+qmodes # Number of dimensions
totalpoints=2**10#2**16 # Total number of points in grid
npoints = np.zeros(dim,dtype=np.int16) # Initialize array
for ii in range(dim):
  npoints[ii] = int(np.sqrt(totalpoints)) # 16 # Number of grid divisions in each direction

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

# Initialize potential parameters
init_pot(xshbt,yshbt,pot_surf='gs',pot_type='full') # Initialize ground state
init_pot(xshbt,yshbt,pot_surf='es',pot_type='full') # Initialize excited state
displacements=bath_displacement(0.,0.,'gs')
coupling=bath_coupling(0.,0.,'gs')
smallcoupling=np.zeros(qmodes,dtype=float)
for ii in range(qmodes):
  smallcoupling[ii]=coupling[ii,ii]
  
# Define grid lengths and define maximum and minimum values of x and y axes in data file
L = np.ones((dim))
for ii in range(qmodes):
  L[ii+2]=20.*np.sqrt(1./np.sqrt(smallcoupling[ii]))
minpos = np.zeros(dim,dtype=np.float) # Initialize array
minpos[0] = -100 # Minimum value of x axis
minpos[1] = -250 # Minimum value of y axis
for i in range(qmodes):
  minpos[i+2]= displacements[i]-L[i+2]/2 # -L[i]/2
maxpos = np.zeros(2,dtype=np.float) # Initialize array
maxpos[0] = 100 # 70. # Maximum value of x axis
maxpos[1] = 250 # 260. # Maximum value of y axis
L[0] = maxpos[0]-minpos[0]
L[1] = maxpos[1]-minpos[1]

# Calculate grid division size
dx = np.zeros(dim,dtype=np.float) # Initialize array
for ii in range(dim):
  dx[ii] = L[ii]/(npoints[ii]-1.)#(maxpos[ii]-minpos[ii])/npoints[ii]

# Calculate grids
xaxis = np.arange(minpos[0],maxpos[0]+dx[0]/2,dx[0]) # Make x-axis for plot
yaxis = np.arange(minpos[1],maxpos[1]+dx[1]/2,dx[1]) # Make y-axis for plot
zaxis=[np.arange(minpos[ii+2],minpos[ii+2]+L[ii+2]+dx[ii+2],dx[ii+2]) for ii in range(qmodes)]
xmesh, ymesh = np.meshgrid(xaxis, yaxis) # Make mesh grids for plot

# Calculate interpolated grid-based potential
# Define ground state potential matrix
potentialground=np.zeros((npoints[0],npoints[1]),dtype=float) # Initialize array
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    potentialground[ii,jj]=reactive_potential(xaxis[ii],yaxis[jj],pot_surf='gs')

# Define excited state potential matrix
potentialexcited=np.zeros((npoints[0],npoints[1]),dtype=float) # Initialize array
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    potentialexcited[ii,jj]=reactive_potential(xaxis[ii],yaxis[jj],pot_surf='es')

# Plot potentials
levelsground = np.linspace(np.min(potentialground),np.max(potentialground),100) # Make ground levels
levelsexcited = np.linspace(np.min(potentialexcited),np.max(potentialexcited),100) # Make exc. levels
nlines = 5 # Number of sharp contour lines
fig, ax = plt.subplots(2,sharex=True,sharey=True) # Create subplots environment
cs1 = ax[0].contourf(xaxis,yaxis,np.transpose(potentialground),levels=levelsground, \
  cmap='jet',extend='max') # Plot ground state
cl1 = ax[0].contour(cs1,levels=cs1.levels[::nlines],colors='k',linewidths=1.) # Add sharp contour lines
cs2 = ax[1].contourf(xaxis,yaxis,np.transpose(potentialexcited),levels=levelsexcited, \
  cmap='jet',extend='max') # Plot excited state
cl2 = ax[1].contour(cs2,levels=cs2.levels[::nlines],colors='k',linewidths=1.) # Add sharp contour lines
fig.suptitle('Interpolated Potentials') # Add figure title
ax[0].set_title('Ground State') # Add first subplot title
ax[1].set_title('Excited State') # Add second subplot title
ax[1].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add joint x-axis label
ax[0].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add first y-axis label
ax[1].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add second y-axis label
plt.show()

# Generate a tensor train of the potential function
# Make tensor train for position space
xone = np.zeros(totalpoints,dtype=float) # Initialize array
oney = np.zeros(totalpoints,dtype=float) # Initialize array
onez = [np.zeros(totalpoints,dtype=float) for ii in range(qmodes)] # assumes all grid dimensions equal
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    index = ii+npoints[0]*jj # Write index for position
    xone[index] = xaxis[jj] # Save x positions into pseudo-rows
    oney[index] = yaxis[ii] # Save y positions into pseudo-columns
    for kk in range(qmodes):
      onez[kk][index]=zaxis[kk][ii]
justone = tt.ones(npoints[0]) # Save tensor train of ones (Note: this assumes
  # there are the same number of grid points in all directions)
ttxone = tt.tensor(np.reshape(xone,[npoints[0],npoints[1]])) # Save xone in tt form
ttoney = tt.tensor(np.reshape(oney,[npoints[0],npoints[1]])) # Save oney in tt form
ttonez=[tt.tensor(np.reshape(onez[ii],[npoints[0],npoints[1]])) for ii in range(qmodes)] # Assumes number of grid points equal in all dimensions
rones = tt.ones(npoints[0],1) # Assumes number of grid points equal in all dimensions
tt_x = [genlist(rones,i,dim,ttxone,ttoney,ttonez) for i in range(dim)]# Generate
  # list of positions in dim dimensions

# Make tensor trains for potentials
eps = 1e-14 # Define accuracy parameter
rma = 10 # Define maximum rank
tt_groundstate = tt.multifuncrs(tt_x, ttground, eps,verb=0,rmax=rma) # Use cross
  # approximation to save the ground state as a tensor train
tt_excitedstate = tt.multifuncrs(tt_x, ttexcited, eps,verb=0,rmax=rma) # Use cross
  # approximation to save the excited state as a tensor train

# Plot tensor-train approximation of 2D potentials
levelsground = np.linspace(np.min(potentialground),np.max(potentialground),100) # Make ground levels
levelsexcited = np.linspace(np.min(potentialexcited),np.max(potentialexcited),100) # Make exc. levels
nlines = 5 # Number of sharp contour lines
fig, ax = plt.subplots(2,sharex=True,sharey=True) # Create subplots environment
cs1 = ax[0].contourf(xaxis,yaxis,np.transpose(tt_groundstate.full()),levels=levelsground,cmap='jet',extend='max') # Plot ground state
cl1 = ax[0].contour(cs1,levels=cs1.levels[::nlines],colors='k',linewidths=1.) # Add sharp contour lines
cs2 = ax[1].contourf(xaxis,yaxis,np.transpose(tt_excitedstate.full()),levels=levelsexcited,cmap='jet',extend='max') # Plot excited state
cl2 = ax[1].contour(cs2,levels=cs2.levels[::nlines],colors='k',linewidths=1.) # Add sharp contour lines
fig.suptitle('Approximated Potentials') # Add figure title
ax[0].set_title('Ground State') # Add first subplot title
ax[1].set_title('Excited State') # Add second subplot title
ax[1].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add joint x-axis label
ax[0].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add first y-axis label
ax[1].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add second y-axis label
plt.show()

# Approximate ground state
# Find the global minimum of the ground state potential
groundglobalmin = np.amin(potentialground) # Determine the global minimum value
groundglobalminpos = np.where(potentialground == groundglobalmin) # Determine the global minimizers
print("Ground state minimum value: ",groundglobalmin)
groundglobalxindex = groundglobalminpos[0][0] # Save the global minimizer indices
groundglobalyindex = groundglobalminpos[1][0]
print("Ground state minimum indices: ",groundglobalxindex,groundglobalyindex)
groundglobalxpos = xaxis[groundglobalxindex] # Save the global minimizer positions
groundglobalypos = yaxis[groundglobalyindex]
print("Ground state minimum position: ",groundglobalxpos,groundglobalypos)

# Find the global minimum of the excited state potential
excitedglobalmin = np.amin(potentialexcited) # Determine the global minimum value
excitedglobalminpos = np.where(potentialexcited == excitedglobalmin) # Determine the global minimizers
print("Excited state minimum value: ",excitedglobalmin)
excitedglobalxindex = excitedglobalminpos[0][0] # Save the global minimizer indices
excitedglobalyindex = excitedglobalminpos[1][0]
print("Excited state minimum indices: ",excitedglobalxindex,excitedglobalyindex)
excitedglobalxpos = xaxis[excitedglobalxindex] # Save the global minimizer positions
excitedglobalypos = yaxis[excitedglobalyindex]
print("Excited state minimum position: ",excitedglobalxpos,excitedglobalypos)

# Find the global minima of the left and right wells of the excited state potential
potentialexcitedleft=1e10*np.ones((npoints[0],npoints[1]),dtype=float) # Initialize
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    if xaxis[ii] < 0:
      potentialexcitedleft[ii,jj]=potentialexcited[ii,jj] # Save left potential
excitedleftglobalmin = np.amin(potentialexcitedleft) # Determine the global minimum value
excitedleftglobalminpos = np.where(potentialexcitedleft == excitedleftglobalmin) # Determine the left potential global minimizers
print("Excited state left minimum value: ",excitedleftglobalmin)
excitedleftglobalxindex = excitedleftglobalminpos[0][0] # Save the global minimizer indices
excitedleftglobalyindex = excitedleftglobalminpos[1][0]
print("Excited state left minimum indices: ",excitedleftglobalxindex,excitedleftglobalyindex)
excitedleftglobalxpos = xaxis[excitedleftglobalxindex] # Save the global minimizer positions
excitedleftglobalypos = yaxis[excitedleftglobalyindex]
print("Excited state left minimum position: ",excitedleftglobalxpos,excitedleftglobalypos)

potentialexcitedright=1e10*np.ones((npoints[0],npoints[1]),dtype=float) # Initialize
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    if xaxis[ii] > 0:
      potentialexcitedright[ii,jj]=potentialexcited[ii,jj] # Save right potential
excitedrightglobalmin = np.amin(potentialexcitedright) # Determine the global minimum value
excitedrightglobalminpos = np.where(potentialexcitedright == excitedrightglobalmin) # Determine the right potential global minimizers
print("Excited state right minimum value: ",excitedrightglobalmin)
excitedrightglobalxindex = excitedrightglobalminpos[0][0] # Save the global minimizer indices
excitedrightglobalyindex = excitedrightglobalminpos[1][0]
print("Excited state right minimum indices: ",excitedrightglobalxindex,excitedrightglobalyindex)
excitedrightglobalxpos = xaxis[excitedrightglobalxindex] # Save the global minimizer positions
excitedrightglobalypos = yaxis[excitedrightglobalyindex]
print("Excited state right minimum position: ",excitedrightglobalxpos,excitedrightglobalypos)

# Calculate the width of a Gaussian at the global minimum of the ground state potential
harmapproxx = (potentialground[groundglobalxindex+1,groundglobalyindex]-2*potentialground [groundglobalxindex,groundglobalyindex]+potentialground[groundglobalxindex-1,groundglobalyindex])/(dx[0])**2 # Calculate the harmonic approximation in x
harmapproxy = (potentialground[groundglobalxindex,groundglobalyindex+1]-2*potentialground[groundglobalxindex,groundglobalyindex]+potentialground[groundglobalxindex,groundglobalyindex-1])/(dx[1])**2 # Calculate the harmonic approximation in y
print("Ground state harmonic approximation: ",harmapproxx,harmapproxy)
sigmax = np.sqrt(2)/harmapproxx**(0.25) # Calculate the sigma parameter in x
sigmay = np.sqrt(2)/harmapproxy**(0.25) # Calculate the sigma parameter in y
print("Ground state approx. sigma: ",sigmax,sigmay)
freqx = np.sqrt(harmapproxx) # Calculate x frequency
freqy = np.sqrt(harmapproxy) # Calculate y frequency
au2cmn1=2.19475e5 # Atomic units to wavenumbers
print("Ground state frequencies [cm-1]: ",freqx*au2cmn1,freqy*au2cmn1)
zpex = freqx/2 # Calculate x ZPE
zpey = freqy/2 # Calculate y ZPE
zpetotal = zpex+zpey # Calculate total ZPE of large-amplitude modes
print("Ground state large-amplitude mode ZPE (x, y, total, total [cm-1]): ",zpex,zpey,zpetotal,zpetotal*au2cmn1)

# Calculate the width of a Gaussian at the global minimum of the excited state potential
excitedharmapproxx = (potentialexcited[excitedglobalxindex+1,excitedglobalyindex]-2*potentialexcited [excitedglobalxindex,excitedglobalyindex]+potentialexcited[excitedglobalxindex-1,excitedglobalyindex])/(dx[0])**2 # Calculate the harmonic approximation in x
excitedharmapproxy = (potentialexcited[excitedglobalxindex,excitedglobalyindex+1]-2*potentialexcited[excitedglobalxindex,excitedglobalyindex]+potentialexcited[excitedglobalxindex,excitedglobalyindex-1])/(dx[1])**2 # Calculate the harmonic approximation in y
print("Excited state harmonic approximation: ",excitedharmapproxx,excitedharmapproxy)
excitedsigmax = np.sqrt(2)/excitedharmapproxx**(0.25) # Calculate the sigma parameter in x
excitedsigmay = np.sqrt(2)/excitedharmapproxy**(0.25) # Calculate the sigma parameter in y
print("Excited state approx. sigma: ",excitedsigmax,excitedsigmay)
excitedfreqx = np.sqrt(excitedharmapproxx) # Calculate x frequency
excitedfreqy = np.sqrt(excitedharmapproxy) # Calculate y frequency
au2cmn1=2.19475e5 # Atomic units to wavenumbers
print("Excited state frequencies [in cm-1]: ",excitedfreqx*au2cmn1,excitedfreqy*au2cmn1)
excitedzpex = excitedfreqx/2 # Calculate x ZPE
excitedzpey = excitedfreqy/2 # Calculate y ZPE
excitedzpetotal = excitedzpex+excitedzpey # Calculate total ZPE of large-amplitude modes
print("Excited state large-amplitude mode ZPE (x, y, total, total [cm-1]): ",excitedzpex,excitedzpey,excitedzpetotal,excitedzpetotal*au2cmn1)

# Calculate the width of a Gaussian at the global minimum of the excited state potential left and right wells
excitedleftharmapproxx = (potentialexcitedleft[excitedleftglobalxindex+1,excitedleftglobalyindex]-2*potentialexcitedleft [excitedleftglobalxindex,excitedleftglobalyindex]+potentialexcitedleft[excitedleftglobalxindex-1,excitedleftglobalyindex])/(dx[0])**2 # Calculate the harmonic approximation in x
excitedleftharmapproxy = (potentialexcitedleft[excitedleftglobalxindex,excitedleftglobalyindex+1]-2*potentialexcitedleft[excitedleftglobalxindex,excitedleftglobalyindex]+potentialexcitedleft[excitedleftglobalxindex,excitedleftglobalyindex-1])/(dx[1])**2 # Calculate the harmonic approximation in y
print("Excited state left harmonic approximation: ",excitedleftharmapproxx,excitedleftharmapproxy)
excitedleftsigmax = np.sqrt(2)/excitedleftharmapproxx**(0.25) # Calculate the sigma parameter in x
excitedleftsigmay = np.sqrt(2)/excitedleftharmapproxy**(0.25) # Calculate the sigma parameter in y
print("Excited state left approx. sigma: ",excitedleftsigmax,excitedleftsigmay)
excitedleftfreqx = np.sqrt(excitedleftharmapproxx) # Calculate x frequency
excitedleftfreqy = np.sqrt(excitedleftharmapproxy) # Calculate y frequency
print("Excited state left frequencies [in cm-1]: ",excitedleftfreqx*au2cmn1,excitedleftfreqy*au2cmn1)
excitedleftzpex = excitedleftfreqx/2 # Calculate x ZPE
excitedleftzpey = excitedleftfreqy/2 # Calculate y ZPE
excitedleftzpetotal = excitedleftzpex+excitedleftzpey # Calculate total ZPE of large-amplitude modes
print("Excited state left large-amplitude mode ZPE (x, y, total, total [cm-1]): ",excitedleftzpex,excitedleftzpey,excitedleftzpetotal,excitedleftzpetotal*au2cmn1)

excitedrightharmapproxx = (potentialexcitedright[excitedrightglobalxindex+1,excitedrightglobalyindex]-2*potentialexcitedright [excitedrightglobalxindex,excitedrightglobalyindex]+potentialexcitedright[excitedrightglobalxindex-1,excitedrightglobalyindex])/(dx[0])**2 # Calculate the harmonic approximation in x
excitedrightharmapproxy = (potentialexcitedright[excitedrightglobalxindex,excitedrightglobalyindex+1]-2*potentialexcitedright[excitedrightglobalxindex,excitedrightglobalyindex]+potentialexcitedright[excitedrightglobalxindex,excitedrightglobalyindex-1])/(dx[1])**2 # Calculate the harmonic approximation in y
print("Excited state right harmonic approximation: ",excitedrightharmapproxx,excitedrightharmapproxy)
excitedrightsigmax = np.sqrt(2)/excitedrightharmapproxx**(0.25) # Calculate the sigma parameter in x
excitedrightsigmay = np.sqrt(2)/excitedrightharmapproxy**(0.25) # Calculate the sigma parameter in y
print("Excited state right approx. sigma: ",excitedrightsigmax,excitedrightsigmay)
excitedrightfreqx = np.sqrt(excitedrightharmapproxx) # Calculate x frequency
excitedrightfreqy = np.sqrt(excitedrightharmapproxy) # Calculate y frequency
print("Excited state right frequencies [in cm-1]: ",excitedrightfreqx*au2cmn1,excitedrightfreqy*au2cmn1)
excitedrightzpex = excitedrightfreqx/2 # Calculate x ZPE
excitedrightzpey = excitedrightfreqy/2 # Calculate y ZPE
excitedrightzpetotal = excitedrightzpex+excitedrightzpey # Calculate total ZPE of large-amplitude modes
print("Excited state right large-amplitude mode ZPE (x, y, total, total [cm-1]): ",excitedrightzpex,excitedrightzpey,excitedrightzpetotal,excitedrightzpetotal*au2cmn1)

# Visualize a bath mode displacement as a function of the large amplitude modes
specmode=6 # Mode of interest

# Make tensor trains for displacements
tt_displacements = tt.multifuncrs(tt_x, ttdisplacements, eps,verb=0,rmax=rma) # Use
  # cross approximation to save the displacements as a tensor train
tt_displacements=tt_displacements.round(eps,rma)

# Define displacements matrix
specdisplacements=np.zeros((npoints[0],npoints[1]),dtype=float) # Initialize array
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    roundx=xaxis[ii]
    roundy=yaxis[jj]
    if roundx < xminorig:
      roundx=xminorig
    elif roundx > xmaxorig:
      roundx=xmaxorig
    if roundy < yminorig:
      roundy=yminorig
    elif roundy > ymaxorig:
      roundy=ymaxorig
    displacements=bath_displacement(roundx,roundy,'es')
    specdisplacements[ii,jj]=displacements[specmode]

# Visualize a bath mode coupling as a function of the large amplitude modes (full Hessian)
specmode0=0
specmode1=1 # Mode of interest

# Make tensor trains for couplings
tt_couplings = tt.multifuncrs(tt_x, ttcouplingsfull, eps,verb=0,rmax=rma) # Use
  # cross approximation to save the displacements as a tensor train
tt_couplings=tt_couplings.round(eps,rma)

# Define displacements matrix
speccouplings=np.zeros((npoints[0],npoints[1]),dtype=float) # Initialize array
for ii in range(npoints[0]):
  for jj in range(npoints[1]):
    roundx=xaxis[ii]
    roundy=yaxis[jj]
    if roundx < xminorig:
      roundx=xminorig
    elif roundx > xmaxorig:
      roundx=xmaxorig
    if roundy < yminorig:
      roundy=yminorig
    elif roundy > ymaxorig:
      roundy=ymaxorig
    couplings=bath_coupling(roundx,roundy,'es')
    speccouplings[ii,jj]=couplings[specmode0,specmode1]
    
# Plot tensor-train displacements
levelsdisplacement = np.linspace(np.min(specdisplacements),np.max(specdisplacements),100) # Make displacement levels
nlines = 5 # Number of sharp contour lines
fig, ax = plt.subplots(1,2,figsize=(12,4),sharey=True) # Create subplots environment
cs2 = ax[0].imshow(np.transpose(tt_displacements.full()),cmap='jet',extent=[minpos[0], maxpos[0],minpos[1],maxpos[1]],aspect=0.25) # Plot displacements
ax[0].set_title('Approximate Displacements for Mode No. %i'%(specmode+1)) # Add figure title
ax[0].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add joint x-axis label
ax[0].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add first y-axis label

# Plot tensor-train couplings
levelscoupling = np.linspace(np.min(speccouplings),np.max(speccouplings),100) # Make displacement levels
nlines = 5 # Number of sharp contour lines
cs2 = ax[1].imshow(np.transpose(tt_couplings.full()),cmap='jet',extent=[minpos[0], maxpos[0],minpos[1],maxpos[1]],aspect=0.25) # Plot displacements # Plot approximate displacements
ax[1].set_title('Approximate Couplings Between Modes No. %i '%(specmode0+1) + 'and No. %i'%(specmode1+1)) # Add figure title
ax[1].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add first y-axis label
plt.savefig('ApproximateParamters.png')
plt.show()

# Plot grid-based displacements
levelsdisplacement = np.linspace(np.min(specdisplacements),np.max(specdisplacements),100) # Make displacement levels
nlines = 5 # Number of sharp contour lines
fig, ax = plt.subplots(1,2,figsize=(12,4),sharey=True) # Create subplots environment
cs2 = ax[0].imshow(np.transpose(specdisplacements),cmap='jet',extent=[minpos[0], maxpos[0],minpos[1],maxpos[1]],aspect=0.25) # Plot displacements
ax[0].set_title('Displacements for Mode No. %i'%(specmode+1)) # Add figure title
ax[0].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add joint x-axis label
ax[0].set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add first y-axis label

# Plot grid-based couplings
levelscoupling = np.linspace(np.min(speccouplings),np.max(speccouplings),100) # Make displacement levels
nlines = 5 # Number of sharp contour lines
cs2 = ax[1].imshow(np.transpose(speccouplings),cmap='jet',extent=[minpos[0], maxpos[0],minpos[1],maxpos[1]],aspect=0.25) # Plot couplings
ax[1].set_title('Couplings Between Modes No. %i '%(specmode0+1) + 'and No. %i'%(specmode1+1)) # Add figure title
ax[1].set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add first y-axis label
plt.savefig('ExactParamters.png')
plt.show()
quit()
