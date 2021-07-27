import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from matplotlib import cm,ticker
import tt
from HBT_potential import init_pot,reactive_potential,bath_displacement,bath_coupling

def ttpositions(ii): # Generate a tensor train of the positions
  # Note: This assumes all coordinates except the first coordinate
  # has the same grid points
  global dim
  global ttjustone,ttxone,ttoney
  if ii > 1:
    component = ttoney # Save one by y-axis core
    for jj in range(ii-1):
      component = tt.kron(ttjustone,component) # Make list ending in defined cores
    for jj in range(dim-1-ii):
      component = tt.kron(component,ttjustone) # Make list starting in defined cores
  else:
    if ii == 0:
      component = ttxone # Save x-axis by one core
    else:
      component = ttoney # Save one by y-axis core
    for jj in range(dim-2):
      component = tt.kron(component,ttjustone) # Make list starting in defined core
  return component

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
    
def genpartialint(e,dim,xindex,yindex,nx):
    # generator of tt list of coordinates
    xydelta =  np.zeros(nx[0]*nx[1],dtype=float)
    ind=xindex+nx[0]*yindex
    xydelta[ind]=1.
    ttxydelta=tt.tensor(np.reshape(xydelta,[nx[0],nx[1]]))
    w = ttxydelta
    for j in range(dim-2):
        w = tt.kron(w,e)
    return w
    
def readwavefunction(filename):
    global nstates
    global tt_temp
    tt_tensorread=[]
    tt_tensorload=0*tt_temp[0]
    for ii in range(nstates):
      tempfilename=filename+'state%i' % ii
      tempfilename+='.npy'
      tttensorload=np.load(tempfilename,allow_pickle=True)
      tt_tensorload=tt_tensorload.from_list(tttensorload)
      tt_tensorread.append(tt_tensorload)
    return tt_tensorread
    
global ttjustone,ttxone,ttoney
global tt_temp

# Initialize parameters.
qmodes=3 # Must match qmodes in ttmn.py
dim=qmodes+2
nstates=2
d=5
n=2**d

# Give the grid parameters.
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

Lx = 200  # box size x
Ly = 500 # box size y
L = np.ones((dim))
L[0] = Lx
L[1] = Ly
for ii in range(qmodes):
    L[ii+2]=20.*np.sqrt(1./np.sqrt(coupling[ii]))
nx=np.zeros(dim,dtype=int)         # number of grid points per dimension
dx=np.zeros(dim,dtype=float)       # grid point spacing
minpos=np.zeros(dim,dtype=float) # minimum position
minpos[0]=-L[0]/2.
minpos[1]=-L[1]/2.
for i in range(qmodes):
    minpos[i+2]= displacements[i]-L[i+2]/2 # -L[i]/2
for i in range(dim):
    nx[i] = n                      # number of grid points
    dx[i] = L[i]/(nx[i]-1)             # coord grid spacing
ddx=1.0
for i in range(dim):
    ddx=ddx*dx[i]
ddxmodes=1.0
for ii in range(qmodes):
    ddxmodes=ddxmodes*dx[ii+2]

# Generate a tensor train of the position space and wavefunction.
# Make tensor train for position space.
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
for i in range(nx[0]):
    for j in range(nx[1]):
        ind=i+nx[0]*j
        xone[ind]=xs[j]
        oney[ind]=ys[i]
        for ii in range(qmodes):
          onez[ii][ind]=zs[ii][i]
    # tt ones and zeros
rones = tt.ones(nx[0],1) # Assumes number of grid points equal in all dimensions

    # coord tt_x list
ttxone=tt.tensor(np.reshape(xone,[nx[0],nx[1]]))
ttoney=tt.tensor(np.reshape(oney,[nx[0],nx[1]]))
ttonez=[tt.tensor(np.reshape(onez[ii],[nx[0],nx[1]])) for ii in range(qmodes)] # Assumes number of grid points equal in all dimensions
tt_x = [genlist(rones,i,dim,ttxone,ttoney,ttonez) for i in range(dim)]

# Make tensor train of wavefunction
# Make a tensor train the shape of a wavefunction's single state.
tt_temp=tt_x
tt_wavefunction=readwavefunction('realprop0')

# Fine grid for visualization
differencefactor=8
xsfine=np.arange(minpos[0],minpos[0]+L[0]+dx[0]/differencefactor,dx[0]/differencefactor)
ysfine=np.arange(minpos[1],minpos[1]+L[1]+1.5*dx[1]/differencefactor,dx[1]/differencefactor)
xfine,yfine=np.meshgrid(xsfine, ysfine, sparse=False, indexing='ij')

au2fs = 0.02418884254
tau = 12.5
count=0
ttwfnfine = []
ttsave=np.zeros(6)
for qq in [0,160,320,480,640,799]: # Individual time steps to be visualized
  temptime = qq*au2fs*tau
  ttsave[count] = temptime
  tempfilename='realprop%i' % qq
  tt_wavefunction=readwavefunction(tempfilename)

  print("Working on state: ",count+1," of ",6)

  ttwfn=np.zeros((nx[0],nx[1]),dtype=complex)
  for ii in range(nx[0]):
    for jj in range(nx[1]):
      print("Working on indices: ",ii,jj)
      densityvalue=0.
      for ll in range(nstates):
        ttprod=genpartialint(rones,dim,jj,ii,nx)*tt_wavefunction[ll]
        densityvalue=densityvalue+tt.dot(tt_wavefunction[ll],ttprod)*ddxmodes
      ttwfn[ii,jj]=densityvalue
      
  # Interpolate surface
  ff = interp2d(xs,ys,np.abs(ttwfn),kind='cubic')
  ttwfnfine.append(ff(xsfine,ysfine))
  count=count+1

#  ttwfn=ttwfn/np.sqrt(np.sum(abs(ttwfn)**2))
fig=plt.figure(figsize=(12,8))

ax = fig.add_subplot(231, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[0],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[0]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])

ax = fig.add_subplot(232, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[1],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[1]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])

ax = fig.add_subplot(233, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[2],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[2]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])

ax = fig.add_subplot(234, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[3],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[3]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])

ax = fig.add_subplot(235, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[4],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[4]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])

ax = fig.add_subplot(236, projection='3d') # Add plot
c0 = ax.plot_surface(xfine,yfine,ttwfnfine[5],cmap='jet',alpha=0.8,linewidth=0,rstride=1,cstride=1,antialiased=False) # Plot wavefunctions
ax.set_title('Time = %1.2f fs' % ttsave[5]) # Add first subplot title
ax.set_xlabel(r'$\mathregular{Q_1 \ / \ au}$') # Add x-axis label
ax.set_ylabel(r'$\mathregular{Q_2 \ / \ au}$') # Add y-axis label
ax.set_zlabel(r'$\left|\psi\right|^2$') # Add z-axis label
ax.set_zlim(0,0.0004)
ax.set_xticks([-100,0,100])
ax.set_yticks([-200,0,200])
ax.set_zticks([0,0.0004])
#  ax.set_zlim(0,2.5e-13)
plt.savefig('snapshots'+str(dim)+'D.png')
plt.close()
  
quit()
