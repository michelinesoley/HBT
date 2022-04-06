#!/usr/bin/python

import sys
import numpy as np
from math import floor
from read_gauss_fchk import read_natoms,read_pos,read_atmass,read_freq,read_nm
from read_gauss_fchk import remove_com,remove_rot_pos,remove_rot_nm
from read_gauss_fchk import make_xyz
from read_gauss_fchk import au2ang,amu2au,rad2deg

###################################################################
### Reads normal modes of TS 
#FIXME



#fchk of 'freq' Gaussian 16 frequency calculation for 	### 
### the transition state (TS) and equilibrium state (GS) and	###
### gets mass-weighted normal-modes, reorient molecule		###
### and prepare files for normal mode analysis.			###
### If optioned, makes movies of the normal-modes.		###
###								###
### Note: it is VERY IMPORTANT that the labeling of the atoms 	###
### in the TS and GS configuration is the same. 		###

###################################################################
### Reads fchk of 'freq' Gaussian 16 frequency calculation for 	### 
### the transition state (TS) and equilibrium state (GS) and	###
### gets mass-weighted normal-modes, reorient molecule		###
### and prepare files for normal mode analysis.			###
### If optioned, makes movies of the normal-modes.		###
###								###
### Note: it is VERY IMPORTANT that the labeling of the atoms 	###
### in the TS and GS configuration is the same. 		###
###################################################################


###########################################################################
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
### define TS and GS files

inputfile_ts = '../HBT_ts_tdwb97xd_pvtz_optfreq.fchk'
#inputfile_gs = '../../HBT_keto_tdwb97xd_pvtz_optfreq.fchk'
inputfile_gs = '../HBT_enol_tdwb97xd_pvtz_optfreq.fchk'

print('inputfile_ts = ',inputfile_ts)
print('inputfile_gs = ',inputfile_gs)

###################################################################
### read natoms and define some parameters

natoms_ts = read_natoms(inputfile_ts)
natoms_gs = read_natoms(inputfile_gs)

if (natoms_ts != natoms_gs):
	print('Error. Problems with natoms!!!',natoms_ts,natoms_gs)
	sys.exit(1)

natoms = natoms_gs
ndgree = 3*natoms
nfreq = ndgree - 6

print('natoms = ',natoms)
print('ndgree = ',ndgree)
print('nfreq = ',nfreq)

###################################################################
### read positions from Gaussian 16 fchk file 
### Note1: positions are read in Bohr
### Note2: order of positions data is x1,y1,z1,x2,y2,z2,...

pos0=read_pos(inputfile_ts,ndgree)
#pos1=read_pos(inputfile_gs,ndgree)

###################################################################
### read atomic mass from Gaussian 16 fchk file 
### Note1: atomic masses are read in amu
### Note2: order of atomic mass data is atom1,atom2,atom3,...

atmass0=read_atmass(inputfile_ts,natoms)
#atmass1=read_atmass(inputfile_gs,natoms)

#for i in range(natoms):
#	if (atmass0[i] != atmass1[i]):
#		sys.exit('Error. Order of atoms is not the same!!!!!')

### change units of mass (amu --> au)
atmass0 *= amu2au
#atmass1 *= amu2au

###################################################################
### read frequencies from Gaussian 16 fchk file
### Note1: frequencies read in cm^{-1}
### Note2: order of freq data is freq1,freq2,freq3,...

freq0=read_freq(inputfile_ts,nfreq)
#freq1=read_freq(inputfile_gs,nfreq)

###################################################################
### read normal modes from Gaussian 16 fchk file
### Note1: normal modes read in Bohr
### Note2: order of normal mode data is nm[x1,freq1],nm[y1,freq1],nm[z1,freq1],...

nm0=read_nm(inputfile_ts,ndgree,nfreq)
#nm1=read_nm(inputfile_gs,ndgree,nfreq)

###################################################################
### remove translations and rotations

theta0 = 0.0
psi0 = 0.0
phi0 = 0.0
theta1 = 0.0
psi1 = 0.0
phi1 = 0.0

theta0,psi0,phi0 = remove_rot_pos(pos0,idx_0,idx_x,idx_y)
#theta1,psi1,phi1 = remove_rot_pos(pos1,idx_0,idx_x,idx_y)

remove_rot_nm(nm0,theta0,psi0,phi0)
#remove_rot_nm(nm1,theta1,psi1,phi1)

print('Euler0 = ',theta0*rad2deg,psi0*rad2deg,phi0*rad2deg)
#print('Euler1 = ',theta1*rad2deg,psi1*rad2deg,phi1*rad2deg)

###################################################################
### remove COM

x0_cm = 0.0
y0_cm = 0.0
z0_cm = 0.0
x1_cm = 0.0
y1_cm = 0.0
z1_cm = 0.0

x0_cm,y0_cm,z0_cm = remove_com(pos0,atmass0)
#x1_cm,y1_cm,z1_cm = remove_com(pos1,atmass1)

print('com_0 = ',x0_cm,y0_cm,z0_cm)
print('com_1 = ',x1_cm,y1_cm,z1_cm)

###################################################################
### generate movie

make_xyz('pos0.xyz',pos0,atmass0)
#make_xyz('pos1.xyz',pos1,atmass1)

###################################################################
### mass-weigth coordinates

for i in range(ndgree):
	idx_mass = int(floor(i/3))
	pos0[i] *= atmass0[idx_mass]**(0.5)
#	pos1[i] *= atmass1[idx_mass]**(0.5)

###################################################################
### mass-weigth normal modes

for i in range(ndgree):
	for j in range(nfreq):
		idx_mass = int(floor(i/3))
		nm0[i,j] *= atmass0[idx_mass]**(0.5)
#		nm1[i,j] *= atmass1[idx_mass]**(0.5)

###################################################################
### normalize normal modes

for i in range(nfreq):
	norm0 = np.dot(nm0[:,i],nm0[:,i])
#	norm1 = np.dot(nm1[:,i],nm1[:,i])

	nm0[:,i] /= norm0**(0.5)
#	nm1[:,i] /= norm1**(0.5)

###################################################################
### check orthogonality of normal modes

for i in range(nfreq):
	for j in range(nfreq):

		norm0 = np.dot(nm0[:,i],nm0[:,j])
#		norm1 = np.dot(nm1[:,i],nm1[:,j])

		if(abs(norm0) > 0.00001 and i!=j ):
			print('Error in orthogonality of modes # ',i,j,'norm0 = ',norm0)
#		if(abs(norm1) > 0.00001 and i!=j ):
#			print('Error in orthogonality of modes # ',i,j,'norm1 = ',norm1)

###################################################################
### save variables for scan

np.savetxt('pos_ts.dat',pos0)
np.savetxt('atmass_ts.dat',atmass0)
np.savetxt('nm_ts.dat',nm0)
np.savetxt('freq_ts.dat',freq0)

#np.savetxt('pos_gs.dat',pos1)
#np.savetxt('atmass_gs.dat',atmass1)
#np.savetxt('nm_gs.dat',nm1)
#np.savetxt('freq_gs.dat',freq1)

###################################################################
### end program

print('DONE!')

sys.exit(0)

