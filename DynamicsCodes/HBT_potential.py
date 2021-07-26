#!/usr/bin/python

###################################################################
### HBT potential and forces for a given set of NM coordinates	###
###								### 
### The potential is defined in terms of a 2D reactive surface, ###
### characterized by 2 reactive normal mode coordinates. 	###
### The rest of NM coordinates are defined as a quadratic 	###
### expansion around the 2D PES, with harmonic parameters 	###
### depending on the reactive coordinates. 			###
###								###
### Both ground state ('gs') and excite state ('es') potential 	###
### are supported [pot_surf parameter].				###
###								###
### Different approximations, including full potential ('full')	###
### , diagonal bath ('diag'), diagonal bath with fixed 		###
### frequency at (x,y)=(0,0) ('diag_fix'), and uncoupled bath 	###
### with fixed frequency at (x,y)=(0,0) ('harm') are supported	###
### [pot_type parameter].					###
###								###
### Note: all data is in atomic units				###
###								###
### Versions: 							###
###	> 0.3: include bath_displacement function		###
###	       include bath_coupling function			###
###	       include compute_freq function			###
###	> 0.2: allow to initialize both potentials		###
###	       allow a slightly shifted grid			###
###	       no default pot_surf parameter			###
### 	> 0.1: initial release					###
###################################################################

import numpy as np
import sys,os
from matplotlib import pyplot as plt

Ha2cm  = 219474.63068                           # Hartree to cm-1 factor

grid_tol = 1.e-7	# controls the grid tolerance

'''
potfile_es  = '2d_pes_es.dat'
nmfile_es   = '2d_nm_ampl_es.dat'
hessfile_es = '2d_hess_ampl_es.dat'
diagfile_es = '2d_hess_diag_es.dat'
potfile_gs  = '2d_pes_gs.dat'
nmfile_gs   = '2d_nm_ampl_gs.dat'
hessfile_gs = '2d_hess_ampl_gs.dat'
diagfile_gs = '2d_hess_diag_gs.dat'
'''

potfile_es  = '2d_pes_es_interpolated.dat'
nmfile_es   = '2d_nm_ampl_es_interpolated.dat'
hessfile_es = '2d_hess_ampl_es_interpolated.dat'
diagfile_es = '2d_hess_diag_es_interpolated.dat'
potfile_gs  = '2d_pes_gs_interpolated.dat'
nmfile_gs   = '2d_nm_ampl_gs_interpolated.dat'
hessfile_gs = '2d_hess_ampl_gs_interpolated.dat'
diagfile_gs = '2d_hess_diag_gs_interpolated.dat'

#------------------------------------------------------------------
def check_grid(xgrid,ygrid):
	''' check if reactive potential is defined in 2d grid. 

	    input: > xgrid: 1d array with grid points along x axis
	           > ygrid: 1d array with grid points along y axis
	'''

	global potfile,nmfile,hessfile,diagfile

	if(xgrid.ndim!=1 or ygrid.ndim!=1):
		sys.exit('\n [check_grid] Error. xgrid/ygrid should be a 1D array.\n')

	if(not os.path.isfile(potfile)):
		sys.exit('\n [init_pot] Error. potfile "{}" not found.\n'.format(potfile))
	if(not os.path.isfile(nmfile)):
		sys.exit('\n [init_pot] Error. nmfile "{}" not found.\n'.format(nmfile))
	if(not os.path.isfile(hessfile)):
		sys.exit('\n [init_pot] Error. hessfile "{}" not found.\n'.format(hessfile))
	if(not os.path.isfile(diagfile)):
		sys.exit('\n [init_pot] Error. diagfile "{}" not found.\n'.format(diagfile))

	### get dimension and range of grid
	xdim = len(xgrid)
	ydim = len(ygrid)
	xmin = xgrid[0]
	xmax = xgrid[-1]
	ymin = ygrid[0]
	ymax = ygrid[-1]

	print('Checking if potential is defined in 2D grid...')

	### get dimension and range where 2D potential is defined
	header = open(potfile,'r').readline()
	xdim1  = int(header.split('npoints =')[1].split()[0])
	ydim1  = int(header.split('npoints =')[1].split()[1])
	xmin1  = float(header.split('q1_range =')[1].split()[0])
	xmax1  = float(header.split('q1_range =')[1].split()[1])
	ymin1  = float(header.split('q2_range =')[1].split()[0])
	ymax1  = float(header.split('q2_range =')[1].split()[1])
	### check grid consistency
	dxmin = abs(xmin-xmin1) > grid_tol 
	dxmax = abs(xmax-xmax1) > grid_tol 
	dymin = abs(ymin-ymin1) > grid_tol
	dymax = abs(ymax-ymax1) > grid_tol

#	print(xmin,xmax,ymin,ymax)
#	print(xmin1,xmax1,ymin1,ymax1)
#	print(xdim,ydim,xdim1,ydim1)
#	print(dxmin,dymin,dxmax,dymax)

	if(xdim!=xdim1 or dxmin or dxmax or ydim!=ydim1 or dymin or dymax):
		sys.exit('\n [check_grid] Error. 2D potential not defined for grid.\n')

	### get dimension and range where bath amplitudes are defined
	header = open(nmfile,'r').readline()
	xdim1  = int(header.split('npoints =')[1].split()[0])
	ydim1  = int(header.split('npoints =')[1].split()[1])
	xmin1  = float(header.split('q1_range =')[1].split()[0])
	xmax1  = float(header.split('q1_range =')[1].split()[1])
	ymin1  = float(header.split('q2_range =')[1].split()[0])
	ymax1  = float(header.split('q2_range =')[1].split()[1])
	### check grid consistency
	dxmin = abs(xmin-xmin1) > grid_tol 
	dxmax = abs(xmax-xmax1) > grid_tol 
	dymin = abs(ymin-ymin1) > grid_tol
	dymax = abs(ymax-ymax1) > grid_tol
	if(xdim!=xdim1 or dxmin or dxmax or ydim!=ydim1 or dymin or dymax):
		sys.exit('\n [check_grid] Error. Bath amplitudes not defined for grid.\n')

	### get dimension and range where bath hessians are defined
	header = open(hessfile,'r').readline()
	xdim1  = int(header.split('npoints =')[1].split()[0])
	ydim1  = int(header.split('npoints =')[1].split()[1])
	xmin1  = float(header.split('q1_range =')[1].split()[0])
	xmax1  = float(header.split('q1_range =')[1].split()[1])
	ymin1  = float(header.split('q2_range =')[1].split()[0])
	ymax1  = float(header.split('q2_range =')[1].split()[1])
	### check grid consistency
	dxmin = abs(xmin-xmin1) > grid_tol 
	dxmax = abs(xmax-xmax1) > grid_tol 
	dymin = abs(ymin-ymin1) > grid_tol
	dymax = abs(ymax-ymax1) > grid_tol
	if(xdim!=xdim1 or dxmin or dxmax or ydim!=ydim1 or dymin or dymax):
		sys.exit('\n [check_grid] Error. Bath hessians not defined for grid.\n')

	### get dimension and range where bath diagonal hessians are defined
	header = open(diagfile,'r').readline()
	xdim1  = int(header.split('npoints =')[1].split()[0])
	ydim1  = int(header.split('npoints =')[1].split()[1])
	xmin1  = float(header.split('q1_range =')[1].split()[0])
	xmax1  = float(header.split('q1_range =')[1].split()[1])
	ymin1  = float(header.split('q2_range =')[1].split()[0])
	ymax1  = float(header.split('q2_range =')[1].split()[1])
	### check grid consistency
	dxmin = abs(xmin-xmin1) > grid_tol 
	dxmax = abs(xmax-xmax1) > grid_tol 
	dymin = abs(ymin-ymin1) > grid_tol
	dymax = abs(ymax-ymax1) > grid_tol
	if(xdim!=xdim1 or dxmin or dxmax or ydim!=ydim1 or dymin or dymax):
		sys.exit('\n [check_grid] Error. Bath diagonal hessians not defined for grid.\n')

	print('Checking if potential is defined in 2D grid...PASS')
	return
#------------------------------------------------------------------

#------------------------------------------------------------------
def init_pot(xgrid,ygrid,pot_surf,pot_type='full'):
	''' initialize potential variables 

	    input: > xgrid: 1d array with grid points along x axis
	           > ygrid: 1d array with grid points along y axis
	           > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential
	           > pot_type: string indicating type of potential
		      >> 'full': full quantum potential   
		      >> 'diag': diagonal bath modes
		      >> 'diag_fix': diagonal bath modes with fixed frequency at (x,y)=(0,0)
		      >> 'harm': uncoupled bath with fixed frequency at (x,y)=(0,0)

	    output: no output is generated but the following global variables are initialized:
		   > q1/q2: reactive coordinates [array of dim xdim/ydim]
		   > pot: reactive potential [array of dim (xdim,ydim)] 
		   > nm_0: bath amplitudes [array of dim (xdim,ydim,nmodes)]  
		   > hess: bath hessians [array of dim (xdim,ydim,nmodes,nmodes)]
	'''

	global pot_es,nm_0_es,hess_es
	global pot_gs,nm_0_gs,hess_gs
	global q1,q2
	global potfile,nmfile,hessfile,diagfile

	if(xgrid.ndim!=1 or ygrid.ndim!=1):
		sys.exit('\n [init_pot] Error. xgrid/ygrid should be a 1D array.\n')

	print('Initializing potential variables...')
	if(pot_surf=='es'):
		print('using excited state potential')
		potfile  = potfile_es
		nmfile   = nmfile_es
		hessfile = hessfile_es
		diagfile = diagfile_es
	elif(pot_surf=='gs'):
		print('using ground state potential')
		potfile  = potfile_gs
		nmfile   = nmfile_gs
		hessfile = hessfile_gs
		diagfile = diagfile_gs
	else:
		sys.exit('\n [init_pot] Error. pot_surf should be: "es" or "gs".\n')
	
	if(pot_type=='full'):
		print('using full potential')
	elif(pot_type=='diag'):
		print('using diagonal bath approximation')
	elif(pot_type=='diag_fix'):
		print('using diagonal bath with fixed freq approximation')
	elif(pot_type=='harm'):
		print('using uncoupled bath with fixed freq approximation')
	else:
		sys.exit('\n [init_pot] Error. pot_type should be: "full", "diag", "diag_fix" or "harm".\n')

	### check if potential is defined for grid
	check_grid(xgrid,ygrid)

	xdim = len(xgrid)
	ydim = len(ygrid)

	###########################################################
	### load 2D reactive potential and coordinates

	### load data
	data=np.loadtxt(potfile)
	### get q1 and q2 coordinates from first two columns
	q1 = data[0::ydim,0] 
	q2 = data[0:ydim,1] 
	### remove two first columns (q1 and q2 coordinates)
	data = np.delete(data,0,axis=1)
	data = np.delete(data,0,axis=1)
	### reshape pot
	pot = np.reshape(data,[xdim,ydim])

	if(pot_surf=='es'):
		pot_es = np.copy(pot)
	elif(pot_surf=='gs'):
		pot_gs = np.copy(pot)
	del pot
	###########################################################

	###########################################################
	### load nm amplitudes

	### get number of nm
	header=open(nmfile,'r').readline()
	nmodes = int(header.split('nmodes =')[1].split()[0])

	if(pot_type=='harm'):	# no amplitude 
		nm_0 = np.zeros(nmodes)
	else:
		### load data
		data=np.loadtxt(nmfile)
		### remove two first columns (q1 and q2 coordinates)
		data = np.delete(data,0,axis=1)
		data = np.delete(data,0,axis=1)
		### reshape data
		nm_0 = np.reshape(data,[xdim,ydim,nmodes])

	if(pot_surf=='es'):
		nm_0_es = np.copy(nm_0)
	elif(pot_surf=='gs'):
		nm_0_gs = np.copy(nm_0)
	del nm_0
	###########################################################

	###########################################################
	### load nm hessian

	if(pot_type=='diag_fix' or pot_type=='harm'): # diagonal hessian with constant values at (x,y)=(0,0)
		### load data 
		data=np.loadtxt(diagfile)
		### remove two first columns (q1 and q2 coordinates)
		data = np.delete(data,0,axis=1)
		data = np.delete(data,0,axis=1)
		### reshape data
		hess_temp = np.reshape(data,[xdim,ydim,nmodes])
		### apply approximations
		hess = np.zeros(nmodes)
		idx = np.abs(q1 - 0.0).argmin()
		jdx = np.abs(q2 - 0.0).argmin()
		for i in range(nmodes):
			hess[i] = hess_temp[idx,jdx,i]
	elif(pot_type=='diag'): # diagonal hessian
		### load data 
		data=np.loadtxt(diagfile)
		### remove two first columns (q1 and q2 coordinates)
		data = np.delete(data,0,axis=1)
		data = np.delete(data,0,axis=1)
		### reshape data
		hess = np.reshape(data,[xdim,ydim,nmodes])
	else: # full hessian
		### load data
		data=np.loadtxt(hessfile)
		### remove two first columns (q1 and q2 coordinates)
		data = np.delete(data,0,axis=1)
		data = np.delete(data,0,axis=1)
		### reshape data
		nhess = len(np.transpose(data))
		hess_temp = np.reshape(data,[xdim,ydim,nhess])
		### transform from lower tridiagonal to matrix form 
		hess = np.zeros([xdim,ydim,nmodes,nmodes])
		for i in range(xdim):
			for j in range(ydim):
				icount = 0
				for k in range(nmodes):
					for l in range(k+1):
						hess[i,j,k,l] = hess_temp[i,j,icount]
						hess[i,j,l,k] = hess_temp[i,j,icount]
						icount += 1

	if(pot_surf=='es'):
		hess_es = np.copy(hess)
	elif(pot_surf=='gs'):
		hess_gs = np.copy(hess)
	del hess
	###########################################################

	print('Initializing potential variables...done')
	return 
#------------------------------------------------------------------

#------------------------------------------------------------------
def reactive_potential(x,y,pot_surf):
	''' reactive 2D potential energy at (x,y)

	    input:  > x,y: coordinates of the reactive modes
	            > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential
	    output: > returns reactive potential at (x,y) [float number]  
	'''

	global pot_es
	global pot_gs
	global q1,q2

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [reactive_potential] Error. x/y should be a real number.\n')

	if(pot_surf=='es'):
		pot = pot_es
	elif(pot_surf=='gs'):
		pot = pot_gs
	else:
		sys.exit('\n [reactive_potential] Error. pot_surf should be: "es" or "gs".\n')

	idx = np.abs(q1 - x).argmin()
	jdx = np.abs(q2 - y).argmin()

	return float(pot[idx,jdx])
#------------------------------------------------------------------

#------------------------------------------------------------------
def bath_potential(x,y,r,pot_surf):
	''' bath potential energy at (x,y,r)
	    
	    input: > x,y: coordinates of the reactive modes.
	           > r: coordinates of the bath. Should be array of dim nmodes.
	           > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential

	    output: > returns bath potential at (x,y,r) [float number]  

	'''

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [bath_potential] Error. x/y should be a real number.\n')

	### get nm displacement for x,y
	nm_xy = bath_displacement(x,y,pot_surf)

	if(len(r) != nm_xy.shape[-1]):
		sys.exit('\n [bath_potential] Error. r should be an array of dim nmodes.\n')

	### get nm hessian for x,y
	hess_xy = bath_coupling(x,y,pot_surf)

	### compute bath energy
	dx = r - nm_xy
	if(len(hess_xy.shape)==2):
		temp = np.dot(dx,np.matmul(hess_xy,dx))
	elif(len(hess_xy.shape)==1):
		temp = np.dot(hess_xy,np.power(dx,2))

	return 0.5*temp
#------------------------------------------------------------------

#------------------------------------------------------------------
def bath_forces(x,y,r,pot_surf):
	''' forces on bath modes at (x,y,r)
	    
	    input: > x,y: coordinates of the reactive modes.
	           > r: coordinates of the bath. Should be array of dim nmodes.
	           > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential


	    output: > returns forces on bath at (x,y,r) as array of dim nmodes.

	'''

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [bath_forces] Error. x/y should be a real number.\n')

	### get nm displacement for x,y
	nm_xy = bath_displacement(x,y,pot_surf)

	if(len(r) != nm_xy.shape[-1]):
		sys.exit('\n [bath_potential] Error. r should be an array of dim nmodes.\n')

	### get nm hessian for x,y
	hess_xy = bath_coupling(x,y,pot_surf)

	### compute bath force
	dx = r - nm_xy
	if(len(hess_xy.shape)==2):
		temp = np.matmul(hess_xy,dx)
	elif(len(hess_xy.shape)==1):
		temp = hess_xy*dx

	return -temp 
#------------------------------------------------------------------

#------------------------------------------------------------------
def compute_freq(x,y,r0,pot_surf,bath_modes=True):
	''' compute harmonic frequencies at (x,y,r) using finite differences
	    
	    input: > x,y: coordinates of the reactive modes.
	           > r0: coordinates of the bath. Should be array of dim nmodes.
	           > pot_surf: string indicating potential surface.
		      >> 'es': excited state potential
		      >> 'gs': ground state potential
	           > bath_modes: string indicating if bath modes are included.  
		      >> 'True': include bath modes
		      >> 'False': exclude bath modes

	    output: > returns harmonic freq at (x,y,r) [array of dim 2 or nmodes+2]  

	'''

	global nm_0_es
	global nm_0_gs
	global q1,q2

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [compute_freq] Error. x/y should be a real number.\n')

	if(pot_surf=='es'):
		nm_0 = nm_0_es
	elif(pot_surf=='gs'):
		nm_0 = nm_0_gs
	else:
		sys.exit('\n [compute_freq] Error. pot_surf should be: "es" or "gs".\n')

	if(bath_modes):
		if(len(r0) != nm_0.shape[-1]):
			sys.exit('\n [compute_freq] Error. r0 should be an array of dim nmodes.\n')

	if(bath_modes):
		nmodes= nm_0.shape[-1]
		hess = np.zeros([nmodes+2,nmodes+2])
	else:
		hess = np.zeros([2,2])

	### steps
	dq1 = q1[1]-q1[0]
	dq2 = q2[1]-q2[0]
	dr  = 0.01

	### index of q1,q2
	idx = np.abs(q1 - x).argmin()
	jdx = np.abs(q2 - y).argmin()

	############################################################
	### compute dV(q1,q2)/(dq1*dq2) term 

	f0 = reactive_potential(q1[idx  ],q2[jdx  ],pot_surf=pot_surf)
	f1 = reactive_potential(q1[idx+1],q2[jdx  ],pot_surf=pot_surf)
	f2 = reactive_potential(q1[idx-1],q2[jdx  ],pot_surf=pot_surf)
	f3 = reactive_potential(q1[idx  ],q2[jdx+1],pot_surf=pot_surf)
	f4 = reactive_potential(q1[idx  ],q2[jdx-1],pot_surf=pot_surf)
	f5 = reactive_potential(q1[idx+1],q2[jdx+1],pot_surf=pot_surf)
	f6 = reactive_potential(q1[idx-1],q2[jdx+1],pot_surf=pot_surf)
	f7 = reactive_potential(q1[idx+1],q2[jdx-1],pot_surf=pot_surf)
	f8 = reactive_potential(q1[idx-1],q2[jdx-1],pot_surf=pot_surf)

	hess[0,0] += (f1+f2-2.*f0)/dq1**2
	hess[1,1] += (f3+f4-2.*f0)/dq2**2
	hess[1,0] += (f5-f7+f8-f6)/(4.*dq1*dq2)
	hess[0,1] += (f5-f7+f8-f6)/(4.*dq1*dq2)

	if(not bath_modes):
		############################################################
		### diagonilize hessian and compute freq 
		w,v = np.linalg.eig(hess)
		w = np.sort(w)
		fct = np.sign(w)
		w = fct*np.sqrt(fct*w)
		return w

	############################################################
	### compute dV_bath/(dr1*dr2) term 

	for i in range(nmodes):
		for j in range(nmodes):
			if(i==j):
				r1 = np.copy(r0)
				r1[i] += dr
				r2 = np.copy(r0)
				r2[i] -= dr

				f0 = bath_potential(q1[idx],q2[jdx],r0,pot_surf=pot_surf)
				f1 = bath_potential(q1[idx],q2[jdx],r1,pot_surf=pot_surf)
				f2 = bath_potential(q1[idx],q2[jdx],r2,pot_surf=pot_surf)

				hess[2+i,2+j] += (f1+f2-2.*f0)/dr**2
			else:
				r5 = np.copy(r0)
				r5[i] += dr
				r5[j] += dr
				r6 = np.copy(r0)
				r6[i] -= dr
				r6[j] += dr
				r7 = np.copy(r0)
				r7[i] += dr
				r7[j] -= dr
				r8 = np.copy(r0)
				r8[i] -= dr
				r8[j] -= dr

				f5 = bath_potential(q1[idx],q2[jdx],r5,pot_surf=pot_surf)
				f6 = bath_potential(q1[idx],q2[jdx],r6,pot_surf=pot_surf)
				f7 = bath_potential(q1[idx],q2[jdx],r7,pot_surf=pot_surf)
				f8 = bath_potential(q1[idx],q2[jdx],r8,pot_surf=pot_surf)

				hess[2+i,2+j] += (f5-f7+f8-f6)/(4.*dr**2)

	############################################################
	### compute dV_bath/(dq1*dr) term 

	for j in range(nmodes): # bath dof
		r1 = np.copy(r0)
		r1[j] += dr  
		r2 = np.copy(r0)
		r2[j] -= dr  

		### idx
		f5 = bath_potential(q1[idx+1],q2[jdx],r1,pot_surf=pot_surf)
		f6 = bath_potential(q1[idx-1],q2[jdx],r1,pot_surf=pot_surf)
		f7 = bath_potential(q1[idx+1],q2[jdx],r2,pot_surf=pot_surf)
		f8 = bath_potential(q1[idx-1],q2[jdx],r2,pot_surf=pot_surf)

		hess[0,2+j] += (f5-f7+f8-f6)/(4.*dq1*dr)
		hess[2+j,0] += (f5-f7+f8-f6)/(4.*dq1*dr)

		### jdx
		f5 = bath_potential(q1[idx],q2[jdx+1],r1,pot_surf=pot_surf)
		f6 = bath_potential(q1[idx],q2[jdx-1],r1,pot_surf=pot_surf)
		f7 = bath_potential(q1[idx],q2[jdx+1],r2,pot_surf=pot_surf)
		f8 = bath_potential(q1[idx],q2[jdx-1],r2,pot_surf=pot_surf)

		hess[1,2+j] += (f5-f7+f8-f6)/(4.*dq2*dr)
		hess[2+j,1] += (f5-f7+f8-f6)/(4.*dq2*dr)

	############################################################
	### compute dV_bath/(dq1*dq2) term 

	f0 = bath_potential(q1[idx  ],q2[jdx  ],r0,pot_surf=pot_surf)
	f1 = bath_potential(q1[idx+1],q2[jdx  ],r0,pot_surf=pot_surf)
	f2 = bath_potential(q1[idx-1],q2[jdx  ],r0,pot_surf=pot_surf)
	f3 = bath_potential(q1[idx  ],q2[jdx+1],r0,pot_surf=pot_surf)
	f4 = bath_potential(q1[idx  ],q2[jdx-1],r0,pot_surf=pot_surf)
	f5 = bath_potential(q1[idx+1],q2[jdx+1],r0,pot_surf=pot_surf)
	f6 = bath_potential(q1[idx-1],q2[jdx+1],r0,pot_surf=pot_surf)
	f7 = bath_potential(q1[idx+1],q2[jdx-1],r0,pot_surf=pot_surf)
	f8 = bath_potential(q1[idx-1],q2[jdx-1],r0,pot_surf=pot_surf)

	hess[0,0] += (f1+f2-2.*f0)/dq1**2
	hess[1,1] += (f3+f4-2.*f0)/dq2**2
	hess[1,0] += (f5-f7+f8-f6)/(4.*dq1*dq2)
	hess[0,1] += (f5-f7+f8-f6)/(4.*dq1*dq2)

	############################################################
	### diagonilize hessian and compute freq 
	w,v = np.linalg.eig(hess)
	w = np.sort(w)
	fct = np.sign(w)
	w = fct*np.sqrt(fct*w)

	return w
#------------------------------------------------------------------

#------------------------------------------------------------------
def bath_displacement(x,y,pot_surf):
	''' bath displacement at (x,y)
	    
	    input: > x,y: coordinates of the reactive modes.
	           > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential

	    output: > returns bath displacement at (x,y) [array of dim nmodes]  

	'''

	global nm_0_es
	global nm_0_gs
	global q1,q2

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [bath_displacement] Error. x/y should be a real number.\n')

	if(pot_surf=='es'):
		nm_0 = nm_0_es
	elif(pot_surf=='gs'):
		nm_0 = nm_0_gs
	else:
		sys.exit('\n [bath_displacement] Error. pot_surf should be: "es" or "gs".\n')

	idx = np.abs(q1 - x).argmin()
	jdx = np.abs(q2 - y).argmin()

	### get nm displacement at (x,y)
	if(len(nm_0.shape)==1):
		nm_xy = nm_0[:]
	elif(len(nm_0.shape)==3):
		nm_xy = nm_0[idx,jdx,:]

	return nm_xy
#------------------------------------------------------------------

#------------------------------------------------------------------
def bath_coupling(x,y,pot_surf):
	''' bath coupling constant at (x,y)
	    
	    input: > x,y: coordinates of the reactive modes.
	           > pot_surf: string indicating potential surface
		      >> 'es': excited state potential
		      >> 'gs': ground state potential

	    output: > returns bath coupling constant at (x,y) [array of dim nmodes or (nmodes,nmodes)]  

	'''

	global hess_es
	global hess_gs
	global q1,q2

	if(not isinstance(x,float) or not isinstance(y,float)):
		sys.exit('\n [bath_coupling] Error. x/y should be a real number.\n')

	if(pot_surf=='es'):
		hess = hess_es
	elif(pot_surf=='gs'):
		hess = hess_gs
	else:
		sys.exit('\n [bath_coupling] Error. pot_surf should be: "es" or "gs".\n')

	idx = np.abs(q1 - x).argmin()
	jdx = np.abs(q2 - y).argmin()

	### get nm hessian for x,y
	if(len(hess.shape)==1):
		hess_xy = hess[:]
	elif(len(hess.shape)==3):
		hess_xy = hess[idx,jdx,:]
	elif(len(hess.shape)==4):
		hess_xy = hess[idx,jdx,:,:]

	return hess_xy
#------------------------------------------------------------------

