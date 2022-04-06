#!/usr/bin/python

import sys
import numpy as np
from math import floor
from read_gauss_fchk import read_natoms,read_pos,read_atmass,read_freq,read_nm
from read_gauss_fchk import remove_com,remove_rot_pos,remove_rot_nm
from read_gauss_fchk import make_xyz
from read_gauss_fchk import amu2au,rad2deg

###################################################################
###              Generates NM files for scan					###
###																###	
### The script reads Gaussian 16 fchk files from a frequency 	###
### calculation of the transition state (TS) and generates 		###
### mass-weighted normal-modes coordinates for normal mode scan	###
### and  analysis.												###
###																###	
### Input needed by the program:								###
### > Gaussian fchk file from 'freq' calculation				###
###																###
### Output:														###
### > pos_ts.dat: position of particles 						### 
### > atmass_ts.dat: atomic mass of particles					###
### > freq_ts.dat: frequency of NM								###
### > nm_ts.dat: NM displacements								###
###																###
### > pos.xyz: snapshot   										###  
###################################################################

###################################################################
### define TS fchk files

inputfile = '../HBT_ts_tdwb97xd_pvtz_optfreq.fchk'

print('inputfile = ',inputfile)

###################################################################
### define origin, x and y axis of internal coordinates for rotations
### Note: the numbering starts at 1

idx_0 = 1
idx_x = 6
idx_y = 2

print('input indexes are:')
print('idx_0 = ',idx_0)
print('idx_x = ',idx_x)
print('idx_y = ',idx_y)

###################################################################
### read natoms and define some parameters

natoms = read_natoms(inputfile)
ndgree = 3*natoms
nfreq = ndgree - 6

print('natoms = ',natoms)
print('ndgree = ',ndgree)
print('nfreq = ',nfreq)

###################################################################
### read positions from Gaussian 16 fchk file 
### Note1: positions are read in Bohr
### Note2: order of positions data is x1,y1,z1,x2,y2,z2,...

pos=read_pos(inputfile,ndgree)

###################################################################
### read atomic mass from Gaussian 16 fchk file 
### Note1: atomic masses are read in amu
### Note2: order of atomic mass data is atom1,atom2,atom3,...

atmass=read_atmass(inputfile,natoms)

### change units of mass (amu --> au)
atmass *= amu2au

###################################################################
### read frequencies from Gaussian 16 fchk file
### Note1: frequencies read in cm^{-1}
### Note2: order of freq data is freq1,freq2,freq3,...

freq=read_freq(inputfile,nfreq)

###################################################################
### read normal modes from Gaussian 16 fchk file
### Note1: normal modes read in Bohr
### Note2: order of normal mode data is nm[x1,freq1],nm[y1,freq1],nm[z1,freq1],...

nm=read_nm(inputfile,ndgree,nfreq)

###################################################################
### remove translations and rotations

theta = 0.0
psi   = 0.0
phi   = 0.0

theta,psi,phi = remove_rot_pos(pos,idx_0,idx_x,idx_y)

remove_rot_nm(nm,theta,psi,phi)

print('Euler = ',theta*rad2deg,psi*rad2deg,phi*rad2deg)

###################################################################
### remove COM

x_cm = 0.0
y_cm = 0.0
z_cm = 0.0

x_cm,y_cm,z_cm = remove_com(pos,atmass)

print('com = ',x_cm,y_cm,z_cm)

###################################################################
### generate snapshot

make_xyz('pos.xyz',pos,atmass)

###################################################################
### mass-weigth coordinates

for i in range(ndgree):
	idx_mass = int(floor(i/3))
	pos[i] *= atmass[idx_mass]**(0.5)

###################################################################
### mass-weigth normal modes

for i in range(ndgree):
	for j in range(nfreq):
		idx_mass = int(floor(i/3))
		nm[i,j] *= atmass[idx_mass]**(0.5)

###################################################################
### normalize normal modes

for i in range(nfreq):
	norm = np.dot(nm[:,i],nm[:,i])

	nm[:,i] /= norm**(0.5)

###################################################################
### check orthogonality of normal modes

for i in range(nfreq):
	for j in range(nfreq):

		norm = np.dot(nm[:,i],nm[:,j])

		if(abs(norm) > 0.00001 and i!=j ):
			print('Error in orthogonality of modes # ',i,j,'norm0 = ',norm0)

###################################################################
### save variables for scan

np.savetxt('pos_ts.dat',pos)
np.savetxt('atmass_ts.dat',atmass)
np.savetxt('nm_ts.dat',nm)
np.savetxt('freq_ts.dat',freq)

###################################################################
### end program

print('DONE!')

sys.exit(0)

