#!/usr/bin/python

import numpy as np
from math import ceil

###################################################################
### A set of functions to read data from Gaussian 16 fchk files	###
###################################################################

###################################################################
### define conversion factors

au2ang = .529177210903							# au to Angstrom factor (CODATA 2018)
au2cm = 219474.63068 							# au to cm-1 factor
amu2au = 1.67262192369E-27/9.1093837015E-31 	# proton mass over electron mass (CODATA)
rad2deg = 180./np.pi 							# radians to degree factor

###################################################################
### read number of atoms from Gaussian 16 fchk file 

def read_natoms(filename):

	print('reading number of atoms from ',filename)

	lines=open(filename,'r').readlines()

	line_label='Number of atoms'

	for i in range(len(lines)):
		if(line_label in lines[i]):		# found section 
			natoms = int(lines[i].split()[4])
	return natoms

###################################################################
### read positions from Gaussian 16 fchk file 
### Note1: positions are read in Bohr
### Note2: order of positions data is x1,y1,z1,x2,y2,z2,...
### Note3: Gaussian saves positions in row of 5

def read_pos(filename,ndim):

	print('reading position from ',filename)

	lines=open(filename,'r').readlines()

	line_label='Current cartesian coordinates'

	n_lines=int(ceil(float(ndim)/5))

	pos = np.zeros(ndim)

	ncount = 0

	for i in range(len(lines)):
		if(line_label in lines[i]):		# found section 
			offset = i + 1  		# header lines
			for j in range(n_lines):
				for k in range(5):
					pos[ncount] = float(lines[offset+j].split()[k])
					ncount += 1

					if(ncount == ndim):
						break
			break
	return pos

###################################################################
### read atomic mass from Gaussian 16 fchk file 
### Note1: atomic masses are read in amu
### Note2: order of atomic mass data is atom1,atom2,atom3,...
### Note3: Gaussian saves atomic numbers in row of 5

def read_atmass(filename,ndim):

	print('reading atomic mass from ',filename)

	lines=open(filename,'r').readlines()

	line_label='Vib-AtMass'

	n_lines=int(ceil(float(ndim)/5))

	atmass = np.zeros(ndim)

	ncount = 0

	for i in range(len(lines)):
		if(line_label in lines[i]):		# found section 
			offset = i + 1  		# header lines
			for j in range(n_lines):
				for k in range(5):
					atmass[ncount] = float(lines[offset+j].split()[k])
					ncount += 1

					if(ncount == ndim):
						break
			break
	return atmass

###################################################################
### read frequencies from Gaussian 16 fchk file 
### Note1: frequencies read in cm^{-1}
### Note2: order of freq data is freq1,freq2,freq3,...
### Note3: Gaussian saves freq in row of 5

def read_freq(filename,ndim):

	print('reading frequencies from ',filename)  

	lines=open(filename,'r').readlines()

	line_label='Vib-E2'

	n_lines=int(ceil(float(ndim)/5))

	freq = np.zeros(ndim)

	ncount = 0

	for i in range(len(lines)):
		if(line_label in lines[i]):		# found section 
			offset = i + 1  		# header lines
			for j in range(n_lines):
				for k in range(5):
					freq[ncount] = float(lines[offset+j].split()[k])
					ncount += 1

					if(ncount == ndim):
						break
			break
	return freq

###################################################################
### read normal modes from Gaussian 16 fchk file 
### Note1: normal modes read in Bohr
### Note2: order of normal mode data is nm[x1,freq1],nm[y1,freq1],nm[z1,freq1],...
### Note3: Gaussian saves normal modes in row of 5

def read_nm(filename,ndim1,ndim2):

	print('reading normal modes from ',filename)

	lines=open(filename,'r').readlines()

	line_label='Vib-Modes'                                

	ndim = ndim1*ndim2

	n_lines=int(ceil(float(ndim)/5))

	nm = np.zeros([ndim1,ndim2])

	ncount = 0
	idx_at = 0
	idx_mode = 0

	for i in range(len(lines)):
		if(line_label in lines[i]):		# found section 
			offset = i + 1  		# header lines
			for j in range(n_lines):
				for k in range(5):

					nm[idx_at,idx_mode] = float(lines[offset+j].split()[k])

					ncount += 1
					idx_at += 1 

					if(ncount%ndim1==0):
						idx_mode += 1
						idx_at = 0

					if(ncount == ndim):
						break
			break
	return nm

###################################################################
### remove COM

def remove_com(data,mass):

	sum_x = 0.0
	sum_y = 0.0
	sum_z = 0.0
	sum_m = 0.0

	ndim = int(len(data)/3)

	for i in range(ndim):
		sum_x += data[3*i+0]*mass[i]
		sum_y += data[3*i+1]*mass[i]
		sum_z += data[3*i+2]*mass[i]
		sum_m += mass[i]

	sum_x /= sum_m
	sum_y /= sum_m
	sum_z /= sum_m

	for i in range(ndim):
		data[3*i+0] -= sum_x
		data[3*i+1] -= sum_y
		data[3*i+2] -= sum_z
	
	return sum_x,sum_y,sum_z

###################################################################
### define Euler rotation matrix (ZXZ convention)

def get_Euler_matrix(theta,psi,phi):

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

###################################################################
### remove translations and rotations from position

def remove_rot_pos(pos,idx_0,idx_x,idx_y):

	ndim = len(pos)

	###########################################################
	### get positions of reference atoms 

	### shift indexes by one (python convention)
	idx_0 -= 1
	idx_x -= 1
	idx_y -= 1

	### origin positions
	r_0 = np.zeros(3)
	r_0[0] = pos[3*idx_0+0]
	r_0[1] = pos[3*idx_0+1]
	r_0[2] = pos[3*idx_0+2]

	### x axis positions
	r_x = np.zeros(3)
	r_x[0] = pos[3*idx_x+0]
	r_x[1] = pos[3*idx_x+1]
	r_x[2] = pos[3*idx_x+2]

	### y axis positions
	r_y = np.zeros(3)
	r_y[0] = pos[3*idx_y+0]
	r_y[1] = pos[3*idx_y+1]
	r_y[2] = pos[3*idx_y+2]

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
		
	R = get_Euler_matrix(theta,psi,phi)

	###########################################################
	### translate positions to origin

	for i in range(int(ndim/3)):
		for j in range(3):
			pos[3*i+j] -= r_0[j]

	###########################################################
	### rotate instanton using Euler angles

	pos_new=np.zeros(ndim)

	for i in range(int(ndim/3)):
		for j in range(3):
			pos_new[3*i+j] = R[0,j] * pos[3*i+0] + R[1,j] * pos[3*i+1] + R[2,j] * pos[3*i+2]

	###########################################################
	### assign new data

	for i in range(ndim):
		pos[i] = pos_new[i]

	return theta,psi,phi

###################################################################
### remove rotations from normal modes

def remove_rot_nm(nm,theta,psi,phi):

	ndim1 = len(nm[:,0])
	ndim2 = len(nm[0,:])

	### get Euler rotation matrix (ZXZ convention)

	R = get_Euler_matrix(theta,psi,phi)

	### rotate nm

	nm_new=np.zeros([ndim1,ndim2])

	for i in range(int(ndim1/3)):
		for j in range(3):
			for k in range(ndim2):
				nm_new[3*i+j,k] = R[0,j] * nm[3*i+0,k] + R[1,j] * nm[3*i+1,k] + R[2,j] * nm[3*i+2,k]

	### assign new data

	for i in range(ndim1):
		for j in range(ndim2):
			nm[i,j] = nm_new[i,j]

	return

###################################################################
### make xyz movie of positions

def make_xyz(filename,data,mass):

	ndim = int(len(data)/3)

	out = open(filename,'w')
	out.write('{} \n'.format(ndim))
	out.write(' \n')
	for i in range(ndim):
		fct_units = au2ang
		x = data[3*i+0] * fct_units
		y = data[3*i+1] * fct_units
		z = data[3*i+2] * fct_units

		if(round(mass[i]/amu2au) == 12):
			label = 'C'	
		elif(round(mass[i]/amu2au) == 1):
			label = 'H'	
		elif(round(mass[i]/amu2au) == 16):
			label = 'O'	
		elif(round(mass[i]/amu2au) == 14):
			label = 'N'	
		elif(round(mass[i]/amu2au) == 32):
			label = 'S'	
		else:
			sys.exit('Error. Label for mass not defined!!!' )

		out.write('{} {} {} {} \n'.format(label,x,y,z))
	out.close()

	return
