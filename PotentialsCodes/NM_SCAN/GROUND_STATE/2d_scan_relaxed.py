#!/usr/bin/python

###################################################################
### Reads normal_modes of TS (previously computed) and 		### 
### generates Gaussian input files for a 2D scan following the	### 
### selected normal modes with other normal modes relaxed 	###
### (employing GIC optimization).				###
###################################################################

import sys
import numpy as np

###################################################################
### define some parameters

nmode1 = 1				# selected normal mode 1  
nmode2 = 5				# selected normal mode 2 

q_max1 = 60.				# maximun Q1 in the scan
q_min1 = -50.				# mimimun Q1 in the scan
dx1 = 10  				# step of scan along Q1

q_max2 = 240.				# maximun Q2 in the scan
q_min2 = -140.				# mimimun Q2 in the scan
dx2 = 20.  				# step of scan along Q2

npoints1 = int((q_max1-q_min1)/dx1)+1	# number of points in the scan along Q1
npoints2 = int((q_max2-q_min2)/dx2)+1	# number of points in the scan along Q2

idebug = 0				# if !=0, print extra info to screen

print('nmode = ',nmode1,nmode2)
print('q_max = ',q_max1,q_max2)
print('q_min = ',q_min1,q_min2)
print('dx = ',dx1,dx2)
print('npoints = ',npoints1,npoints2,npoints1*npoints2)

###################################################################
### define conversion factors

au2ang = .529177210903				# au to Angstrom factor (CODATA 2018)
au2cm  = 219474.63068 				# au to cm-1 factor
amu2au = 1.67262192369E-27/9.1093837015E-31  	# proton mass over electron mass (CODATA)

###################################################################
### load data
### Note1: positions are read in m_e**(0.5)*Bohr
### Note2: masses are read in m_e
### Note3: normal vectors are orthonormal and unitless
### Note4: freq are read in cm-1

path = '../../../../../NM_READER/'

pos = np.loadtxt(path+'pos_ts.dat')

atmass = np.loadtxt(path+'atmass_ts.dat')

nm = np.loadtxt(path+'nm_ts.dat')

freq = np.loadtxt(path+'freq_ts.dat')

if (idebug>0):
	print('freq = ',freq[nmode1-1],freq[nmode2-1])

###################################################################
### define some parameters

ndgree = len(pos)
natoms = int(ndgree/3)

print('ndgree = ',ndgree)
print('natoms = ',natoms)

###################################################################
### generate normal mode scan along selected normal modes

pos_new=np.zeros([npoints1,npoints2,ndgree])

nm_aux1 = nm[:,nmode1-1]
nm_aux2 = nm[:,nmode2-1]

out=open('2d_scan_relaxed.out','w')
out.write('npoints = {} {} {} \n'.format(npoints1,npoints2,npoints1*npoints2))
for i in range(npoints1):
	temp_dx1 = q_min1 + i*dx1
	for j in range(npoints2):
		temp_dx2 = q_min2 + j*dx2

		print('|Q_im| = ',temp_dx1,temp_dx2)
		out.write('|Q_im| = {} {} {} {} \n'.format(temp_dx1,temp_dx2,i+1,j+1))

		Q_im1 = nm_aux1 * temp_dx1

		Q_im2 = nm_aux2 * temp_dx2

		pos_new[i,j,:] = pos + Q_im1 + Q_im2
out.close()

### test COM

if (idebug>0):
	for i in range(npoints1): 
		for j in range(npoints2): 
			sum_x = 0.0
			sum_y = 0.0
			sum_z = 0.0
			sum_m = 0.0
			for k in range(natoms):
				sum_x += pos[3*k+0]*np.power(atmass[k],0.5)
				sum_y += pos[3*k+1]*np.power(atmass[k],0.5)
				sum_z += pos[3*k+2]*np.power(atmass[k],0.5)
				sum_m += atmass[k]

			sum_x /= sum_m
			sum_y /= sum_m
			sum_z /= sum_m

			print('com = ',i,j,sum_x,sum_y,sum_z)

###################################################################
### generate GIC label related options
### Note1: positions are in Bohr
### Note2: masses are in m_e

### fix selected normal modes 1 (labeled 'Q')

nm_aux = nm[:,nmode1-1] 

gic_nm_label1=np.zeros(ndgree,dtype='U300')

for i in range(natoms):
	fct_units = 1./atmass[i]**(0.5)
	x = pos[3*i+0] * fct_units
	y = pos[3*i+1] * fct_units
	z = pos[3*i+2] * fct_units

	mass = atmass[i]

	nm_x = nm_aux[3*i+0]
	nm_y = nm_aux[3*i+1]
	nm_z = nm_aux[3*i+2]

	temp_x = '(X({})-({}))*{}'.format(i+1,x,nm_x) 
	temp_y = '(Y({})-({}))*{}'.format(i+1,y,nm_y) 
	temp_z = '(Z({})-({}))*{}'.format(i+1,z,nm_z) 

	temp_xyz = '({}+{}+{})*SQRT({})'.format(temp_x,temp_y,temp_z,mass)

	gic_nm_label1[i] = 'Q{}(inactive)={}'.format(i+1,temp_xyz)

gic_nm1 = 'Qtot(freeze)='
for j in range(natoms):
	gic_nm1 = gic_nm1 + 'Q{}'.format(j+1)
	if(j!=natoms-1):
		gic_nm1 = gic_nm1 + '+'

### fix selected normal modes 2 (labeled 'P')

nm_aux = nm[:,nmode2-1] 

gic_nm_label2=np.zeros(ndgree,dtype='U300')

for i in range(natoms):
	fct_units = 1./atmass[i]**(0.5)
	x = pos[3*i+0] * fct_units
	y = pos[3*i+1] * fct_units
	z = pos[3*i+2] * fct_units

	mass = atmass[i]

	nm_x = nm_aux[3*i+0]
	nm_y = nm_aux[3*i+1]
	nm_z = nm_aux[3*i+2]

	temp_x = '(X({})-({}))*{}'.format(i+1,x,nm_x) 
	temp_y = '(Y({})-({}))*{}'.format(i+1,y,nm_y) 
	temp_z = '(Z({})-({}))*{}'.format(i+1,z,nm_z) 

	temp_xyz = '({}+{}+{})*SQRT({})'.format(temp_x,temp_y,temp_z,mass)

	gic_nm_label2[i] = 'P{}(inactive)={}'.format(i+1,temp_xyz)

gic_nm2 = 'Ptot(freeze)='
for j in range(natoms):
	gic_nm2 = gic_nm2 + 'P{}'.format(j+1)
	if(j!=natoms-1):
		gic_nm2 = gic_nm2 + '+'

if (idebug>0):
	print('gic_nm_label1 = ',gic_nm_label1)
	print('gic_nm1 = ',gic_nm1)
	print('gic_nm_label2 = ',gic_nm_label2)
	print('gic_nm2 = ',gic_nm2)

###################################################################
### generate Gaussian input files

gjf_header=open('gjf_header.inp','r').readlines()

for i in range(npoints1):
	for j in range(npoints2):

		out = open('config_{}_{}.gjf'.format(i+1,j+1),'w')	

		### write chk option
		out.write('%chk=config_{}_{}.chk \n'.format(i+1,j+1))

		### write header
		for line in gjf_header:
			out.write(line)
	
		### write coordinates
		for k in range(natoms):
			fct_units = au2ang/atmass[k]**(0.5)
			x = pos_new[i,j,3*k]   * fct_units 
			y = pos_new[i,j,3*k+1] * fct_units 
			z = pos_new[i,j,3*k+2] * fct_units

			if(round(atmass[k]/amu2au) == 1):
				label = 'H'	
			elif(round(atmass[k]/amu2au) == 12):
				label = 'C'	
			elif(round(atmass[k]/amu2au) == 14):
				label = 'N'	
			elif(round(atmass[k]/amu2au) == 16):
				label = 'O'	
			elif(round(atmass[k]/amu2au) == 32):
				label = 'S'	
			else:
				sys.exit('Error. Label for mass not defined!!!' )

			out.write('{} {} {} {} \n'.format(label,x,y,z))

		### write extra lines
		out.write(' \n')

		### write inactive GIC options
		for k in range(natoms):
			out.write(gic_nm_label1[k]+'\n')
			out.write(gic_nm_label2[k]+'\n')

		### write active GIC options
		out.write(gic_nm1+'\n')
		out.write(gic_nm2+'\n')

		### write extra lines
		out.write(' \n')
		out.write(' \n')

		### close file
		out.close()

###################################################################
### end program

print('DONE!!')

sys.exit(0)

