#/usr/bin/python

import sys
import numpy as np
from math import ceil,floor
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d

###################################################################
###              Generates Reactive Potential					###
###																###
### The script reads the outputs of Gaussian calculations and 	###
### generates the 2D PES, NM displacement vectors and 			###
### NM coupling matrix as a function of scanned normal modes.	###
###################################################################


Ha2cm  = 219474.63068 				# Hartree to cm-1 factor
a02ang = .529177210903				# Bohr to Angstrom factor (CODATA 2018)
amu2me = 1.67262192369E-27/9.1093837015E-31  	# proton mass over electron mass (CODATA)

###################################################################
### define some parameters

### apply constrain 

#constrain_ts_nm_deriv = True 	# constrain NM displacement derivative at TS?
constrain_ts_nm_deriv = False

print('NM deriv constrain at TS? = ',constrain_ts_nm_deriv)

### define tolerances for setting NM to zero

tol_nm = 1.	# for NM displacements
tol_hess = 35.  # for NM hessian [in cm-1]

tol_hess = (tol_hess/Ha2cm)**2 	#transform from cm-1 to hessian units 

print('tol_nm = ',tol_nm)
print('tol_hess = ',tol_hess)

### path for NM analysis

path = '../../../NM_READ/'

print('NM path = ',path)

### fixed logfiles files

fixed_files=[ [12,1] , [12,4] , [12,5] , [12,8] , [12,9] , [12,10] , [12,12] , [12,13] ]

print('FIXED_FILES = ',fixed_files)

### define origin, x and y axis of internal coordinates for rotations

idx_rot_0 = 1 - 1
idx_rot_x = 6 - 1
idx_rot_y = 2 - 1

print('idx_rot = ',idx_rot_0,idx_rot_x,idx_rot_y)

###################################################################
### read grid dimension

inputfile=open('../2d_scan_relaxed.out','r').readlines()

ndim1 = int(inputfile[0].split()[2])
ndim2 = int(inputfile[0].split()[3])

print('ndim = ',ndim1,ndim2)

###################################################################
### define function to read energy from Gaussian logfiles 
### Note: energy is read in Hartree

energy_gs = np.zeros([ndim1,ndim2])
energy_es = np.zeros([ndim1,ndim2])

energy_gs_error = -1028.
energy_es_error = -1028.

def read_energy(idx,jdx,inputfile):
	''' read energy from Gaussian file '''

	### ---------------------------------------------------------------------------
	### open file
	try:	
		lines=open(inputfile,'r').readlines()
	except IOError:
		print('[reader.py] Error. Logfile # {}_{} not found. Using energy_error = {}/{} as energies'.format(idx+1,jdx+1,energy_gs_error,energy_es_error))
		energy_gs[idx,jdx] = energy_gs_error
		energy_es[idx,jdx] = energy_es_error
		return

	### ---------------------------------------------------------------------------
	### check completion
	line_label = 'Normal termination'

	ierr = 1
	for k in range(len(lines)):
		if(line_label in lines[-k]):  # found normal termination section
			ierr = 0
			break

	if(ierr!=0):
		print('[reader.py] Error. Logfile # {}_{} did not terminate normal.'.format(idx+1,jdx+1))
#		sys.exit(1)

	### ---------------------------------------------------------------------------
	### read GROUND STATE energy
	line_label = 'SCF Done'

	ierr = 1
	for k in range(len(lines)):
		if(line_label in lines[k]):  # found energy section
			energy_gs[idx,jdx] = float(lines[k].split()[4])
			ierr = 0

	if(ierr!=0):
		print('[reader.py] Error. GS energy not found in logfile # {}_{}. Using energy_gs_error = {} as energy'.format(idx+1,jdx+1,energy_gs_error))
		energy_gs[idx,jdx] = energy_gs_error 
#		sys.exit(1)

	### ---------------------------------------------------------------------------
	### read EXCITED STATE energy
	line_label = 'Total Energy, E(TD-HF/TD-DFT)'

	ierr = 1
	for k in range(len(lines)):
		if(line_label in lines[k]):  # found energy section
			energy_es[idx,jdx] = float(lines[k].split()[4])
			ierr = 0

	if(ierr!=0):
		print('[reader.py] Error. ES energy not found in logfile # {}_{}. Using energy_es_error = {} as energy'.format(idx+1,jdx+1,energy_es_error))
		energy_es[idx,jdx] = energy_es_error 
#		sys.exit(1)

	return

###################################################################
### read energy from Gaussian logfiles 

### read energy from config_* files

for i in range(ndim1):
	for j in range(ndim2):
		inputfile='../config_{}_{}.log'.format(i+1,j+1)
		read_energy(i,j,inputfile)

### read energy from '*_fixed' file for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile='../config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	read_energy(idx,jdx,inputfile)

###################################################################
### mask values with error

#energy_gs = np.ma.array(energy_gs, mask=(energy_gs==energy_gs_error))
#energy_es = np.ma.array(energy_es, mask=(energy_es==energy_es_error))

###################################################################
### shift energy and change units

print('energy_min [au] = ',np.min(energy_gs),np.min(energy_es))
print('energy_max [au] = ',np.max(energy_gs),np.max(energy_es))

#energy_gs *= Ha2cm
#energy_es *= Ha2cm

energy_0 = np.min(energy_gs)
energy_1 = np.min(energy_gs)
#energy_1 = np.min(energy_es)

energy_gs -= energy_0
energy_es -= energy_1

print('energy_baseline [au] = ',energy_1-energy_0)

###################################################################
### load NM vectors 
### Note: pos, atmass and nm are in atomic units.

pos0 = np.loadtxt(path+'pos_ts.dat')

atmass = np.loadtxt(path+'atmass_ts.dat')

nm = np.loadtxt(path+'nm_ts.dat')

ndgree = len(pos0)
natoms = int(ndgree/3)
nfreq = ndgree-6
nhess = int((ndgree*ndgree+ndgree)/2)  	# number of data in lower tridiagonal form 

print('ndgree = ',ndgree)
print('natoms = ',natoms)
print('nfreq = ',nfreq)
print('nhess = ',nhess)

###################################################################
### define functions to read positions from Gaussian 16 logfile and compute nm displacement 

def read_pos(filename):
	''' read position from Gaussian 16 logfile '''
	''' Note: positions are read in Angstrom '''

	lines=open(filename,'r').readlines()

	label='Coordinates'

	x=np.zeros(natoms)
	y=np.zeros(natoms)
	z=np.zeros(natoms)

	for i in range(len(lines)):
		if(label in lines[i]):		# found position section 
			offset = i+2  		# header lines
			for j in range(natoms):
				x[j] = float(lines[offset+1+j].split()[3])
				y[j] = float(lines[offset+1+j].split()[4])
				z[j] = float(lines[offset+1+j].split()[5])
	return x,y,z

def mass_weight_pos(x,y,z):
	''' generate mass-weight position vector (in atomic units)'''

	pos1=np.zeros(ndgree)

	for i in range(natoms):
		fct_units = atmass[i]**(0.5)/a02ang
		pos1[3*i+0] = x[i] * fct_units   
		pos1[3*i+1] = y[i] * fct_units
		pos1[3*i+2] = z[i] * fct_units

	return pos1

def nm_displacement(pos1):
	''' compute NM displacement '''

	dx = pos1 - pos0

	suma=np.zeros(nfreq)

	for k in range(nfreq):
		nm_aux = nm[:,k]
		suma[k] = np.dot(dx,nm_aux)

	return suma

def remove_com(data_x,data_y,data_z):
	''' remove center of mass '''

	sum_x = 0.0
	sum_y = 0.0
	sum_z = 0.0
	sum_m = 0.0

	for i in range(natoms):
		sum_x += data_x[i] * atmass[i]
		sum_y += data_y[i] * atmass[i]
		sum_z += data_z[i] * atmass[i]
		sum_m += atmass[i]

	sum_x /= sum_m
	sum_y /= sum_m
	sum_z /= sum_m
	print('COM = ',sum_x,sum_y,sum_z)

	for i in range(natoms):
		data_x[i] -= sum_x 
		data_y[i] -= sum_y 
		data_z[i] -= sum_z 
	
	return data_x,data_y,data_z

def Euler_matrix(theta,psi,phi):
	''' Euler rotation matrix (ZXZ convention) '''

	R = np.zeros([3,3])

	R[0,0] = -np.sin(phi) * np.cos(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)
	R[1,0] =  np.cos(phi) * np.cos(theta) * np.sin(psi) + np.sin(phi) * np.cos(psi)
	R[2,0] =                np.sin(theta) * np.sin(psi)

	R[0,1] = -np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)
	R[1,1] =  np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi)
	R[2,1] =                np.sin(theta) * np.cos(psi)

	R[0,2] =  np.sin(phi) * np.sin(theta)
	R[1,2] = -np.cos(phi) * np.sin(theta)
	R[2,2] =                np.cos(theta)

	return R

def remove_rotation(data_x,data_y,data_z,idx_0,idx_x,idx_y):
	''' remove rotation using references atoms '''
	''' only removing rotation around the plane (i.e. psi=phi=0) '''

	###########################################################
	### get positions of reference atoms 

	### origin positions
	r_0 = np.zeros(3)
	r_0[0] = data_x[idx_0]
	r_0[1] = data_y[idx_0]
	r_0[2] = data_z[idx_0]

	### x axis positions
	r_x = np.zeros(3)
	r_x[0] = data_x[idx_x]
	r_x[1] = data_y[idx_x]
	r_x[2] = data_z[idx_x]

	### y axis positions
	r_y = np.zeros(3)
	r_y[0] = data_x[idx_y]
	r_y[1] = data_y[idx_y]
	r_y[2] = data_z[idx_y]

	###########################################################
	### generate molecular frame vectors a,b,c

	a = np.zeros(3)
	b = np.zeros(3)
	for i in range(3):
	        a[i] = r_x[i] - r_0[i]
	        b[i] = r_y[i] - r_0[i]
	c = np.cross(a,b)
	b = np.cross(c,a) # since b could not be orthonormal to a
	a /= np.linalg.norm(a)
	b /= np.linalg.norm(b)
	c /= np.linalg.norm(c)

	###########################################################
	### get Euler angles and rotation matrix (ZXZ convention)

	theta = np.arccos(c[2])
	if (theta!=0.0):	# psi and phi uniquely defined
		psi = np.arctan2(a[2],b[2])
		phi = np.arctan2(c[0],-c[1])
	else:			# (psi + phi) uniquely defined
		psi = 0.0
		phi = np.arccos(a[0]) 

	# only removing rotation around the plane
	psi = 0.0
	phi = 0.0

	print('Euler = ',theta,psi,phi)

	R = Euler_matrix(theta,psi,phi)

	###########################################################
	### translate positions to origin

	for i in range(natoms):
			data_x[i] -= r_0[0]
			data_y[i] -= r_0[1]
			data_z[i] -= r_0[2]

	###########################################################
	### rotate positions using Euler angles

	data_new_x = np.zeros(natoms)
	data_new_y = np.zeros(natoms)
	data_new_z = np.zeros(natoms)

	for i in range(natoms):
			data_new_x[i] = R[0,0] * data_x[i] + R[1,0] * data_y[i] + R[2,0] * data_z[i]
			data_new_y[i] = R[0,1] * data_x[i] + R[1,1] * data_y[i] + R[2,1] * data_z[i]
			data_new_z[i] = R[0,2] * data_x[i] + R[1,2] * data_y[i] + R[2,2] * data_z[i]

	return data_new_x,data_new_y,data_new_z

def make_xyz(filename,data):
	''' make xyz snapshot file ''' 
	''' Note: input in mass-weighted coordinates in atomic units '''

	out = open(filename,'w')
	out.write('{} \n'.format(natoms))
	out.write(' \n')
	for i in range(natoms):
		fct_units = a02ang/atmass[i]**(0.5)
		x = data[3*i+0] * fct_units
		y = data[3*i+1] * fct_units
		z = data[3*i+2] * fct_units

		if(round(atmass[i]/amu2me) == 12):
			label = 'C'	
		elif(round(atmass[i]/amu2me) == 1):
			label = 'H'	
		elif(round(atmass[i]/amu2me) == 16):
			label = 'O'	
		elif(round(atmass[i]/amu2me) == 14):
			label = 'N'	
		elif(round(atmass[i]/amu2me) == 32):
			label = 'S'	
		else:
			sys.exit('Error. Label for mass not defined!!!' )

		out.write('{} {} {} {} \n'.format(label,x,y,z))
	out.close()

	return

###################################################################
### read positions from Gaussian 16 logfile and compute nm displacement 

nm_ampl = np.zeros([ndim1,ndim2,nfreq])

### loop over Gaussian 16 logfile 

for i in range(ndim1):
	for j in range(ndim2):
		inputfile='../config_{}_{}.log'.format(i+1,j+1)
		x,y,z = read_pos(inputfile)
#		x,y,z = remove_rotation(x,y,z,idx_rot_0,idx_rot_x,idx_rot_y)
#		x,y,z = remove_com(x,y,z)
		pos1 = mass_weight_pos(x,y,z)
		nm_ampl[i,j,:] = nm_displacement(pos1)
#		make_xyz('config_{}_{}.xyz'.format(i+1,j+1),pos1)

### read nm_ampl from '*_fixed' file for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile='../config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	x,y,z = read_pos(inputfile)
#	x,y,z = remove_rotation(x,y,z,idx_rot_0,idx_rot_x,idx_rot_y)
#	x,y,z = remove_com(x,y,z)
	pos1 = mass_weight_pos(x,y,z)
	nm_ampl[idx,jdx,:] = nm_displacement(pos1)

###################################################################
### identify reactive NM using four extreme points as reference

tol = 5.e-5

for i in range(nfreq):
	temp0 = np.abs(nm_ampl[0,0,i] - nm_ampl[0,-1,i])
	temp1 = np.abs(nm_ampl[-1,0,i] - nm_ampl[-1,-1,i])
	temp2 = np.abs(nm_ampl[0,0,i] - nm_ampl[-1,0,i])  
	temp3 = np.abs(nm_ampl[0,-1,i] - nm_ampl[-1,-1,i])  

	### q1
	if(temp0<tol and temp1<tol):
		index_x = i
		print('index_x = ',i+1)

	### q2
	if(temp2<tol and temp3<tol):
		index_y = i
		print('index_y = ',i+1)

### define reactive NM grid 

q1 = nm_ampl[:,0,index_x].round(2)
q2 = nm_ampl[0,:,index_y].round(2)
print('q1 =',q1) 
print('q2 =',q2) 

###################################################################
### set to zero NM with small amplitudes
### Note: some NM should be zero (symmetry consideration) but are not 
###       due to numerical error. 

for i in range(nfreq):
	lmax = np.max(np.abs(nm_ampl[:,:,i]))
	if (lmax<tol_nm):
		nm_ampl[:,:,i] = 0.0

###################################################################
### compute displacement between TS and EQ position for each NM

index_ts1, = np.where(q1==0.0)
index_ts2, = np.where(q2==0.0)
index_ts = [index_ts1,index_ts2]
print('TS index = ',index_ts)

e_min = np.min(energy_es)
index_eq = np.where(energy_es==e_min)
print('MIN1 index = ',index_eq)

nm_ampl_max1 = np.zeros(nfreq)
for i in range(nfreq):
	temp = np.abs(nm_ampl[index_ts[0],index_ts[1],i]-nm_ampl[index_eq[0],index_eq[1],i])
	nm_ampl_max1[i] = temp

e_min = np.min(energy_es[:int(ndim1/2),:])
index_eq = np.where(energy_es==e_min)
print('MIN2 index = ',index_eq)

nm_ampl_max2 = np.zeros(nfreq)
for i in range(nfreq):
	temp = np.abs(nm_ampl[index_ts[0],index_ts[1],i]-nm_ampl[index_eq[0],index_eq[1],i])
	nm_ampl_max2[i] = temp

###################################################################
### define functions to extract hessian information from fchk file 

def read_hess(filename):
	''' read hessian from Gaussian 16 fchk file '''
	''' Note1: force constants are read in atomic units '''
	''' Note2: Gaussian saves force constants in row of 5 '''
	''' Note3: Gaussian saves the hessian in lower tridiagonal form '''

	lines=open(filename,'r').readlines()

	line_label='Cartesian Force Constants'

	n_lines=int(ceil(float(nhess)/5))

	temp = np.zeros(nhess)

	ncount = 0

	for i in range(len(lines)):
		if(line_label in lines[i]):	# found hessian section 
			offset = i + 1  	# header lines
			for j in range(n_lines):
				for k in range(5):
					temp[ncount] = float(lines[offset+j].split()[k])
					ncount += 1

					if(ncount == nhess):
						break
			break
	return temp

def mass_weight_hess(temp):
	''' generate mass-weight hessian '''

	icount = 0
	for i in range(ndgree):
		idx_mass_i = int(floor(i/3))
		for j in range(i+1):
			idx_mass_j = int(floor(j/3))

			temp[icount] /= (atmass[idx_mass_i]*atmass[idx_mass_j])**(0.5)
			icount+=1
	return temp 

def tridiagonal_to_full_hess(temp):
	''' generate full hessian from lower tridiagonal hessian '''

	temp_new = np.zeros([ndgree,ndgree])

	### generate lower tridiagonal matrix 
	icount = 0
	for i in range(ndgree):
		for j in range(i+1):
			temp_new[i,j] = temp[icount]
			icount+=1
	### symmetrize 
	for i in range(ndgree):
		for j in range(i+1):
			temp_new[j,i] = temp_new[i,j]

	return temp_new

def cartesian_to_nm_hess(temp):
	''' generate hessian in NM from hessian in cartesian '''

	### V=HxU matrix multiplication
	temp_new = np.matmul(temp,nm)  

	### U^{*}xV matrix multiplication
	temp_new = np.matmul(np.transpose(nm),temp_new) 

	return temp_new 

def diagonalize(temp):
	''' diagonalize hessian '''
	''' Note: the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]. '''

	eigv,vec = np.linalg.eig(temp)

	return eigv,vec

def delta(idx,jdx):
	''' delta function '''
	if(idx==jdx):
		return float(1)
	else:
		return float(0)

def eps(idx,jdx,kdx):
	''' Levi-Civita function '''
	if(idx==jdx or jdx==kdx or kdx==idx):
		return float(0)
	elif(idx==0 and jdx==1 and kdx==2):
		return float(1)
	elif(idx==1 and jdx==2 and kdx==0):
		return float(1)
	elif(idx==2 and jdx==0 and kdx==1):
		return float(1)
	elif(idx==2 and jdx==1 and kdx==0):
		return float(-1)
	elif(idx==1 and jdx==0 and kdx==2):
		return float(-1)
	elif(idx==0 and jdx==2 and kdx==1):
		return float(-1)
	else:
		print(idx,jdx,kdx)
		sys.exit('Something wrong with Levi-Civita function')

def project_hess(temp,inputfile):
	''' project out translation and rotation from hessian '''
	''' Based on J. Chem. Phys. 81, 3942 (1984) '''

	'''		
	### read positions 
	x,y,z = read_pos(inputfile)

	### compute COM 
	xcom = 0.
	ycom = 0. 
	zcom = 0.
	tot_mass = 0.
	for i in range(natoms):
		xcom += x[i] * atmass[i] 	
		ycom += y[i] * atmass[i] 	
		zcom += z[i] * atmass[i] 	
		tot_mass += atmass[i]
	xcom /= tot_mass
	ycom /= tot_mass
	zcom /= tot_mass

	for i in range(natoms):
		x[i] -= xcom 
		y[i] -= ycom 
		z[i] -= zcom 

	pos = np.zeros([natoms,3])
	for i in range(natoms):
		pos[i,0] = x[i] 
		pos[i,1] = y[i] 
		pos[i,2] = z[i] 

	### translation matrix
	trans_mtrx = np.zeros([ndgree,ndgree])
	for i in range(ndgree):
		idx_mass_i = int(floor(i/3))
		idx_cart_i = i%3
		for j in range(ndgree):
			idx_mass_j = int(floor(j/3))
			idx_cart_j = j%3
			trans_mtrx[i,j] += np.sqrt(atmass[idx_mass_i]*atmass[idx_mass_j])/tot_mass*delta(idx_cart_i,idx_cart_j) 
	'''
	'''	
	### compute inertia tensor
	I0 = np.zeros([3,3])
	for i in range(natoms):
		I0[0,0] += atmass[i]*(y[i]**2+z[i]**2) 
		I0[1,1] += atmass[i]*(x[i]**2+z[i]**2) 
		I0[2,2] += atmass[i]*(x[i]**2+y[i]**2) 
		I0[0,1] -= atmass[i]*x[i]*y[i] 
		I0[0,2] -= atmass[i]*x[i]*z[i] 
		I0[1,2] -= atmass[i]*y[i]*z[i] 

	I0[1,0] = I0[0,1]
	I0[2,0] = I0[0,2]
	I0[2,1] = I0[1,2]

	I0_inv = np.linalg.inv(I0)

	### rotation matrix
	rot_mtrx = np.zeros([ndgree,ndgree])
	 
	for i in range(ndgree):
		idx_mass_i = int(floor(i/3))
		idx_cart_i = i%3
		for j in range(ndgree):
			idx_mass_j = int(floor(j/3))
			idx_cart_j = j%3

			for k in range(3):       # alpha 
			  for l in range(3):     # alpha'
			    for m in range(3):   # beta
			      for n in range(3): # beta'

			        rot_mtrx[i,j] += np.sqrt(atmass[idx_mass_i]*atmass[idx_mass_j])*pos[idx_mass_i,m]*pos[idx_mass_j,n]*I0_inv[k,l]*eps(k,m,idx_cart_i)*eps(l,n,idx_cart_j)
	'''		

	### NM matrix
	nm_aux1 = nm[:,index_x]	
	nm_aux2 = nm[:,index_y]	

	nm_mtrx = np.zeros([ndgree,ndgree])
	for i in range(ndgree):
		for j in range(ndgree):
			nm_mtrx[i,j] += nm_aux1[i]*nm_aux1[j] + nm_aux2[i]*nm_aux2[j]

	### projection matrix
	proj_mtrx = np.identity(ndgree)
	#proj_mtrx -= trans_mtrx
	#proj_mtrx -= rot_mtrx
	proj_mtrx -= nm_mtrx

	### project hessian

	temp_new = np.matmul(temp,proj_mtrx)  
	temp_new = np.matmul(proj_mtrx,temp_new) 

	return temp_new

def print_eigv(data):
	''' print eigenvalues in cm-1 '''

	w = data
	#w = np.sort(w)
	fct = np.sign(w)
	w = fct*np.sqrt(fct*w)
	print('eigv = ',w*Ha2cm)
	return

###################################################################
### constrain hessian normal modes at TS 
### Note1: Redefine NM at TS and compare with old NM to check consistency.
###       If contrain_nm = True, use new NM to compute hessians; otherwise
###       use old NM.  
#FIXME

index_ts1, = np.where(q1==0.0)
index_ts2, = np.where(q2==0.0)
inputfile='../config_{}_{}.fchk'.format(int(index_ts1)+1,int(index_ts2)+1)
print(inputfile)
hess_temp = read_hess(inputfile)
hess_temp = mass_weight_hess(hess_temp)
hess_temp = tridiagonal_to_full_hess(hess_temp)
eigv,evalc = diagonalize(hess_temp)
print_eigv(eigv)

### identify zero freq modes
zero_freq = np.zeros(6)
icount = 0
for i in range(ndgree):
	w = eigv[i]
	fct = np.sign(w)
	w = fct*np.sqrt(fct*w)*Ha2cm
	if(np.abs(w) < 15):
		zero_freq[icount] = i
		icount += 1
print('zero_freq = ',zero_freq)
if (icount!=6):
	sys.exit('Error in zero_freq')

### compare with old NM
### Note: new NM are ordered in ascending freq.
w = np.sort(eigv)
for i in range(6): # delete zero freq modes
	w = np.delete(w,1)

nm_new = np.zeros([ndgree,nfreq])
for i in range(ndgree):
	if(i in zero_freq): #skip zero freq modes
		continue

	idx, = np.where(eigv[i] == w)
	nm_aux1 = nm[:,int(idx)]
	nm_aux2 = evalc[:,i]

	fct = np.dot(nm_aux1,nm_aux2)
	nm_aux2 *= np.sign(fct)
	diff = nm_aux1 - nm_aux2
	err = np.dot(diff,diff)
	#print(i,idx,np.sign(fct),err.round(4))

	if(np.abs(err) > 0.1):
		sys.exit('Error in new vs old NM # {} {}'.format(i,err))

	nm_new[:,int(idx)] = nm_aux2

### overwrite old NM with new NM for computing Hessian
if(constrain_ts_hess):
	nm = np.copy(nm_new)

###################################################################
### read hessian from Gaussian 16 fchk file 

hess_ampl = np.zeros([ndim1,ndim2,nfreq,nfreq])  

### loop over Gaussian 16 logfile 

for i in range(ndim1):
	for j in range(ndim2):

		inputfile='../config_{}_{}.fchk'.format(i+1,j+1)
		print(inputfile)
		hess_temp = read_hess(inputfile)
		hess_temp = mass_weight_hess(hess_temp)
		hess_temp = tridiagonal_to_full_hess(hess_temp)

		inputfile='../config_{}_{}.log'.format(i+1,j+1)
		#print(inputfile)
		hess_temp = project_hess(hess_temp,inputfile)

		hess_temp = cartesian_to_nm_hess(hess_temp)
		hess_ampl[i,j,:,:] = hess_temp

### loop over '*_fixed' files for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile='../config_{}_{}_fixed.fchk'.format(idx+1,jdx+1)
	print(inputfile)
	hess_temp = read_hess(inputfile)
	hess_temp = mass_weight_hess(hess_temp)
	hess_temp = tridiagonal_to_full_hess(hess_temp)

	inputfile='../config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	print(inputfile)
	hess_temp = project_hess(hess_temp,inputfile)

	hess_temp = cartesian_to_nm_hess(hess_temp)
	hess_ampl[idx,jdx,:,:] = hess_temp

###################################################################
### shift non-diagonal hessians using TS as reference
#FIXME

index_ts1, = np.where(q1==0.0)
index_ts2, = np.where(q2==0.0)

hess_ref = hess_ampl[int(index_ts1),int(index_ts2),:,:]

for i in range(nfreq):
	for j in range(nfreq):
		if(i==j):
			continue
#		hess_ampl[:,:,i,j] -= hess_ref[i,j]

###################################################################
### set to zero non-diagonal hessians with small amplitudes

for i in range(nfreq):
	for j in range(nfreq):
		if(i==j):
			continue

#		temp1 = np.abs(hess_ampl[:,:,i,j]) 
#		temp2 = np.abs(hess_ampl[:,:,i,i] - hess_ampl[:,:,j,j])
#		lmax = np.max(temp1/temp2)

		lmax = np.max(np.abs(hess_ampl[:,:,i,j]))

		if (lmax<tol_hess):
			hess_ampl[:,:,i,j] = 0.0

###################################################################
### compute displacement between TS and EQ position for each NM

index_ts1, = np.where(q1==0.0)
index_ts2, = np.where(q2==0.0)
index_ts = [index_ts1,index_ts2]
print('TS index = ',index_ts)

e_min = np.min(energy_es)
index_eq = np.where(energy_es==e_min)
print('MIN1 index = ',index_eq)

hess_ampl_max1 = np.zeros([nfreq,nfreq])
for i in range(nfreq):
	for j in range(nfreq):
		temp = np.abs(hess_ampl[index_ts[0],index_ts[1],i,j]-hess_ampl[index_eq[0],index_eq[1],i,j])
		hess_ampl_max1[i,j] = temp

e_min = np.min(energy_es[:int(ndim1/2),:])
index_eq = np.where(energy_es==e_min)
print('MIN2 index = ',index_eq)

hess_ampl_max2 = np.zeros([nfreq,nfreq])
for i in range(nfreq):
	for j in range(nfreq):
		temp = np.abs(hess_ampl[index_ts[0],index_ts[1],i,j]-hess_ampl[index_eq[0],index_eq[1],i,j])
		hess_ampl_max2[i,j] = temp

###################################################################
### define functions to plot data

### define some options

plt.rcParams['axes.linewidth'] = 1.5

color=['black','white']

lw=1.

labelsz=12.

ticksz=11

mksz = 6

nlines = 2

def plot_2d_2x1(data0,data1,title0='',title1='',xlabel0='',xlabel1='',ylabel0='',ylabel1='',ctitle='',lmax0=0,lmax1=0):

	### define figures

	fig,ax= plt.subplots(2,sharex=True,sharey=True)

	#fig.subplots_adjust(bottom=0.12,right=0.95,top=0.92)

	### plot data

	if(lmax0==0): 
		lmax0 = np.max(data0)
	lmin0 = np.min(data0)
	ldel0 = (lmax0-lmin0)/100
	#print('data0 lmax,lmin,ldel = ',lmax0,lmin0,ldel0)
	levels0 = np.arange(lmin0,lmax0+ldel0,ldel0)

	if(lmax1==0): 
		lmax1 = np.max(data1)
	lmin1 = np.min(data1)
	ldel1 = (lmax1-lmin1)/100
	#print('data1 lmax,lmin,ldel = ',lmax1,lmin1,ldel1)
	levels1 = np.arange(lmin1,lmax1+ldel1,ldel1)

	CS0 = ax[0].contourf(q1,q2,np.transpose(data0),levels=levels0,cmap='jet',extend='max')
	CL0 = ax[0].contour(CS0,levels=CS0.levels[::nlines],colors='k',linewidths=lw)

	CS1 = ax[1].contourf(q1,q2,np.transpose(data1),levels=levels1,cmap='jet',extend='max')
	CL1 = ax[1].contour(CS1,levels=CS1.levels[::nlines],colors='k',linewidths=lw)

	### plot colorbars

	##CS0.cmap.set_under('yellow')
	CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax[0])
	CB0.add_lines(CL0)
	CB0.ax.set_title(ctitle,size=labelsz)
	CB0.set_ticks(np.arange(0,levels0[-1]+levels0[1],ldel0*10))

	##CS1.cmap.set_under('yellow')
	CS1.cmap.set_over('white')
	CB1 = fig.colorbar(CS1,ax=ax[1])
	CB1.add_lines(CL1)
	CB1.ax.set_title(ctitle,size=labelsz)
	CB1.set_ticks(np.arange(0,levels1[-1]+levels1[1],ldel1*10))

	'''
	### mark minimun and TS of EXCITED STATE
	index1, = np.where(q1==0.0)
	index2, = np.where(q2==0.0)
	e_value = energy_es[index1,index2]
	x1 = q1[index1]
	y1 = q2[index2]
	print('x_ts,y_ts,e_value = ',x1,y1,e_value)
	ax[0].plot(x1, y1, color="white", marker = "o",markersize=mksz)

	e_value = np.min(energy_es[1:int(index1),:])  # left minimum
	index = np.where(energy_es==e_value)
	x1 = q1[index[0]]
	y1 = q2[index[1]]
	print('x_min,y_min,e_value = ',x1,y1,e_value)
	ax[0].plot(x1, y1, color="white", marker = "o",markersize=mksz)

	e_value = np.min(energy_es[int(index1):ndim1,:])  # right minimum
	index = np.where(energy_es==e_value)
	x1 = q1[index[0]]
	y1 = q2[index[1]]
	print('x_min,y_min,e_value = ',x1,y1,e_value)
	ax[0].plot(x1, y1, color="white", marker = "o",markersize=mksz)

	### mark minimun of GROUND STATE

	e_value = np.min(energy_gs[1:int(index1),:])  # left minimum
	index = np.where(energy_gs==e_value)
	x1 = q1[index[0]]
	y1 = q2[index[1]]
	print('x_min,y_min,e_value = ',x1,y1,e_value)
	ax[1].plot(x1, y1, color="white", marker = "o",markersize=mksz)

	e_value = np.min(energy_gs[int(index1)+3:ndim1,:])  # right minimum
	index = np.where(energy_gs==e_value)
	x1 = q1[index[0]]
	y1 = q2[index[1]]
	print('x_min,y_min,e_value = ',x1,y1,e_value)
	#ax[1].plot(x1, y1, color="white", marker = "o",markersize=mksz)
	'''

	### set labels

	ax[0].set_title(title0,size=labelsz)
	ax[1].set_title(title1,size=labelsz)

	ax[0].set_xlabel(xlabel0,size=labelsz)
	ax[1].set_xlabel(xlabel1,size=labelsz)
	ax[0].set_ylabel(ylabel0,size=labelsz)
	ax[1].set_ylabel(ylabel1,size=labelsz)

	### set axis limits

	x_axis=[-60,60]
	y_axis=[-125,60]

	#ax.set_xlim(x_axis)
	#ax.set_ylim(y_axis)

	### set axis ticks

	ax[0].tick_params(axis='both',which='major',labelsize=ticksz)
	ax[1].tick_params(axis='both',which='major',labelsize=ticksz)

	### plot legend

	#ax.legend(loc=0,numpoints=1,edgecolor='k')

	### plot

	#plt.show()

	return fig

def plot_2d_1x1(data,title='',xlabel='',ylabel='',ctitle=''):

	### set levels

	lmax = np.max(data)
	lmin = np.min(data)
	ldel = (lmax-lmin)/100
	#print('data lmax,lmin,ldel = ',lmax,lmin,ldel)
	if(ldel==0.0):
		return
	levels = np.arange(lmin,lmax+ldel,ldel)

	### define figures

	fig,ax= plt.subplots()

	#fig.subplots_adjust(bottom=0.12,right=0.95,top=0.92)

	### plot data

	CS0 = ax.contourf(q1,q2,np.transpose(data),levels=levels,cmap='jet',extend='max')
	CL0 = ax.contour(CS0,levels=CS0.levels[::nlines],colors='k',linewidths=lw)

	### plot colorbars

	##CS0.cmap.set_under('yellow')
	CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax)
	CB0.add_lines(CL0)
	CB0.ax.set_title(ctitle,size=labelsz)
	#CB0.set_ticks(np.arange(0,levels[-1]+levels[1],400))

	### set labels

	ax.set_title(title,size=labelsz)

	ax.set_xlabel(xlabel,size=labelsz)
	ax.set_ylabel(ylabel,size=labelsz)

	### set axis limits

	x_axis=[-60,60]
	y_axis=[-125,60]

	#ax.set_xlim(x_axis)
	#ax.set_ylim(y_axis)

	### set axis ticks

	ax.tick_params(axis='both',which='major',labelsize=ticksz)

	### plot legend

	#ax.legend(loc=0,numpoints=1,edgecolor='k')

	### plot

	#plt.show()

	return fig 

def plot_matrix(data,title='',xlabel='',ylabel='',ctitle=''):

	### define figures

	fig,ax= plt.subplots()

	#fig.subplots_adjust(bottom=0.12,right=0.95,top=0.92)

	### plot data

	CS0 = ax.imshow(np.transpose(data),interpolation='none',cmap='jet')

	### plot colorbars

	##CS0.cmap.set_under('yellow')
	CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax)
	CB0.ax.set_title(ctitle,size=labelsz)
	#CB0.set_ticks(np.arange(0,levels[-1]+levels[1],400))

	### set labels

	ax.set_title(title,size=labelsz)

	ax.set_xlabel(xlabel,size=labelsz)
	ax.set_ylabel(ylabel,size=labelsz)

	### set axis limits

	x_axis=[-60,60]
	y_axis=[-125,60]

	#ax.set_xlim(x_axis)
	#ax.set_ylim(y_axis)

	### set axis ticks

	ax.tick_params(axis='both',which='major',labelsize=ticksz)

	### plot legend

	#ax.legend(loc=0,numpoints=1,edgecolor='k')

	### plot

	#plt.show()

	return fig 

def plot_bar(data,title='',xlabel='',ylabel=''):

	### define figures

	figsize=[14,6]

	fig,ax= plt.subplots(figsize=figsize)

	#fig.subplots_adjust(bottom=0.12,right=0.95,top=0.92)

	### plot data

	x = np.arange(0,len(data))

	bars = ax.bar(x,data)

	### annotate labels

	for rect in bars:
		height = rect.get_height()
		center = rect.get_x() + rect.get_width() / 2
		if(height>=1):
			ax.annotate('{}'.format(int(center)),
		                xy=(center, height),
	        	        xytext=(0, 3),  # 3 points vertical offset
	                	textcoords="offset points",
		                ha='center', va='bottom')

	### set labels

	ax.set_title(title,size=labelsz)

	ax.set_xlabel(xlabel,size=labelsz)
	ax.set_ylabel(ylabel,size=labelsz)

	### set axis limits

	x_axis=[-60,60]
	y_axis=[-125,60]

	#ax.set_xlim(x_axis)
	#ax.set_ylim(y_axis)

	### set axis ticks

	ax.tick_params(axis='both',which='major',labelsize=ticksz)

	### plot legend

	#ax.legend(loc=0,numpoints=1,edgecolor='k')

	### plot

	#plt.show()

	return fig 

###################################################################
### plot data

#xlabel = xlabel0 = xlabel1 = r'$\mathregular{Q_{1}}$ / a.u. '
#ylabel = ylabel0 = ylabel1 = r'$\mathregular{Q_{5}}$ / a.u. '
xlabel = xlabel0 = xlabel1 = r'$\mathregular{Q_{PT}}$ / a.u. '
ylabel = ylabel0 = ylabel1 = r'$\mathregular{Q_{Bend}}$ / a.u. '

#---------------------------------------
### PES
data0 = energy_es * Ha2cm
data1 = energy_gs * Ha2cm
title0 = 'Excited state'
title1 = 'Ground state'
ctitle = r'$\mathregular{V \ / \ cm^{-1}}$'

fig1a = plot_2d_2x1(data0,data1,title0=title0,title1=title1,xlabel1=xlabel1,ylabel0=ylabel0,ylabel1=ylabel1,ctitle=ctitle,lmax1=6000)
fig1b = plot_2d_1x1(data0,title=title0,xlabel=xlabel1,ylabel=ylabel0,ctitle=ctitle)

#---------------------------------------
### nm_ampl 
'''
for idx in range(nfreq):
	data = nm_ampl[:,:,idx]
	title = 'NM amplitude # {}'.format(idx+1)
	ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
	plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)
	plt.show()
#plt.show()
'''

idx = 9 - 1
data = nm_ampl[:,:,idx]
title = 'NM amplitude # {}'.format(idx+1)
ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
#fig2a = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

idx = 11 - 1
data = nm_ampl[:,:,idx]
title = 'NM amplitude # {}'.format(idx+1)
ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
#fig2b = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

idx = 61 - 1
data = nm_ampl[:,:,idx]
title = 'NM amplitude # {}'.format(idx+1)
ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
#fig2c = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

idx = 44 - 1
data = nm_ampl[:,:,idx]
title = 'NM amplitude # {}'.format(idx+1)
ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
#fig2d = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

#---------------------------------------
### hess_ampl 

'''
for idx in range(nfreq):
	for jdx in range(nfreq):
		data = hess_ampl[:,:,idx,jdx]
		fct = np.sign(data)
		data = fct*np.power(fct*data,0.5) * Ha2cm
		title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
		ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
		plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)
	plt.show()
'''
'''
for idx in range(nfreq):
	data = hess_ampl[:,:,idx,idx]
	fct = np.sign(data)
	data = fct*np.power(fct*data,0.5) * Ha2cm
	title = 'hess amplitude # {} {}'.format(idx+1,idx+1)
	ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
	plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)
'''
'''
idx=9-1
for jdx in range(nfreq):
	data = hess_ampl[:,:,idx,jdx]
	fct = np.sign(data)
	data = fct*np.power(fct*data,0.5) * Ha2cm
	title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
	ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
	plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)
'''

idx = 11 - 1
jdx = 11 - 1
data = hess_ampl[:,:,idx,jdx]
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
#fig3a = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

idx = 11 - 1
jdx = 9 - 1
data = hess_ampl[:,:,idx,jdx]
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
#fig3b = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

idx = 2 - 1
jdx = 68 - 1
data = hess_ampl[:,:,idx,jdx]
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
#fig3c = plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

#---------------------------------------
### hess_ampl matrix 
'''
for idx in range(ndim1):
	for jdx in range(ndim2):
		data = hess_ampl[idx,jdx,:,:]
		for kdx in range(nfreq):
			data[kdx,kdx] = 0.  #set to zero diagonal elements
		fct = np.sign(data)
		data = fct*np.power(fct*data,0.5) * Ha2cm
		title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
		plot_matrix(data,title=title)
	plt.show()
plt.show()
'''

idx = 6 - 1
jdx = 8 - 1 
data = np.copy(hess_ampl[idx,jdx,:,:])
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
fig4a = plot_matrix(data,title=title)

idx = 6 - 1
jdx = 7 - 1 
data = np.copy(hess_ampl[idx,jdx,:,:])
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
plot_matrix(data,title=title)

idx = 6 - 1
jdx = 9 - 1 
data = np.copy(hess_ampl[idx,jdx,:,:])
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
plot_matrix(data,title=title)

idx = 9 - 1
jdx = 12 - 1 
data = np.copy(hess_ampl[idx,jdx,:,:])
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
fig4b = plot_matrix(data,title=title)

idx = 3 - 1
jdx = 12 - 1 
data = np.copy(hess_ampl[idx,jdx,:,:])
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess amplitude # {} {}'.format(idx+1,jdx+1)
fig4c = plot_matrix(data,title=title)

#---------------------------------------
### maximun coupling matrix

data = np.copy(hess_ampl_max1)
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess MAX1 amplitude'
fig_mtrx1 = plot_matrix(data,title=title)

data = np.copy(hess_ampl_max2)
for i in range(nfreq):
	data[i,i] = 0.  #set to zero diagonal elements
fct = np.sign(data)
data = fct*np.power(fct*data,0.5) * Ha2cm
title = 'hess MAX2 amplitude'
fig_mtrx2 = plot_matrix(data,title=title)

#---------------------------------------
### maximun displacement 

data = nm_ampl_max1
fig_bar1 = plot_bar(data,xlabel='Mode #',ylabel='$\mathregular{|Q_i|}$')

data = nm_ampl_max2
fig_bar2 = plot_bar(data,xlabel='Mode #',ylabel='$\mathregular{|Q_i|}$')

###################################################################
### show plot

#plt.show()

###################################################################
### save plot

fig1a.savefig('2d_pes.pdf',format='pdf',dpi=1200)
fig1b.savefig('2d_pes_es.pdf',format='pdf',dpi=1200)

#fig2a.savefig('nm_ampl_8.pdf',format='pdf',dpi=1200)
#fig2b.savefig('nm_ampl_11.pdf',format='pdf',dpi=1200)
#fig2c.savefig('nm_ampl_61.pdf',format='pdf',dpi=1200)
#fig2d.savefig('nm_ampl_44.pdf',format='pdf',dpi=1200)

#fig3a.savefig('hess_ampl_11_11.pdf',format='pdf',dpi=1200)
#fig3b.savefig('hess_ampl_11_9.pdf',format='pdf',dpi=1200)
#fig3c.savefig('hess_ampl_2_68.pdf',format='pdf',dpi=1200)

fig_bar1.savefig('nm_ampl_max1.pdf',format='pdf',dpi=1200)
fig_bar2.savefig('nm_ampl_max2.pdf',format='pdf',dpi=1200)

fig_mtrx1.savefig('hess_ampl_max1.pdf',format='pdf',dpi=1200)
fig_mtrx2.savefig('hess_ampl_max2.pdf',format='pdf',dpi=1200)

###################################################################
### save data

### PES
out=open('2d_pes_es.dat','w')

out.write('# x (au) y (au) V_es (au) ')
out.write('npoints = {} {} q1_range = {} {} q2_range = {} {} \n'.format(ndim1,ndim2,q1[0],q1[-1],q2[0],q2[-1]))
for i in range(ndim1):
	for j in range(ndim2):
		out.write('{} {} {} \n'.format(q1[i],q2[j],energy_es[i,j]))
out.close()

### NM amplitudes
### Note: removing reactive NM 

nm_ampl = np.delete(nm_ampl,index_x,axis=2)
nm_ampl = np.delete(nm_ampl,index_y-1,axis=2)

out = open('2d_nm_ampl_es.dat','w')

out.write('# x (au) y (au) [Q0_i] ')
out.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1[0],q1[-1],q2[0],q2[-1]))
out.write('nmodes = {} index = {} {} \n'.format(nfreq-2,index_x,index_y))
for i in range(ndim1):
	for j in range(ndim2):
		out.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nfreq-2):
			out.write('{} '.format(nm_ampl[i,j,k]))
		out.write('\n')
out.close()

### Hess amplitudes
### Note: remove reactive NM 
### Note: save in lower tridiagonal form

hess_ampl = np.delete(hess_ampl,index_x,axis=2)
hess_ampl = np.delete(hess_ampl,index_y-1,axis=2)
hess_ampl = np.delete(hess_ampl,index_x,axis=3)
hess_ampl = np.delete(hess_ampl,index_y-1,axis=3)

out = open('2d_hess_ampl_es.dat','w')

out.write('# x (au) y (au) [H_ij] ')
out.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1[0],q1[-1],q2[0],q2[-1]))
out.write('nmodes = {} index = {} {} \n'.format(nfreq-2,index_x,index_y))
for i in range(ndim1):
	for j in range(ndim2):
		out.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nfreq-2):
#			for l in range(nfreq-2):
			for l in range(k+1):
				out.write('{} '.format(hess_ampl[i,j,k,l]))
		out.write('\n')
out.close()

### Diagonal hess amplitudes

out = open('2d_hess_diag_es.dat','w')
out.write('# x (au) y (au) [H_ii] ')
out.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1[0],q1[-1],q2[0],q2[-1]))
out.write('nmodes = {} index = {} {} \n'.format(nfreq-2,index_x,index_y))
for i in range(ndim1):
	for j in range(ndim2):
		out.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nfreq-2):
			out.write('{} '.format(hess_ampl[i,j,k,k]))
		out.write('\n')
out.close()

###################################################################
### end program
print('DONE')
sys.exit(0)


