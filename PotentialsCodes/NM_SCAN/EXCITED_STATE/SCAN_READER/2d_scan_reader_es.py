#/usr/bin/python

import sys
import numpy as np
from math import ceil,floor
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d

###################################################################
###        Generates Reactive Potential Energy Surface			###
###																###
### The script reads the outputs of Gaussian calculations and 	###
### generates the 2D PES, NM displacement vectors and 			###
### NM coupling matrix as a function of scanned normal modes.	###
###																###
### Input needed by the program:								###
### > Output file generated by 2d_scan_relaxed.py 				### 
### > *.log files of Gaussian calculations						###
### > *.fchk files of Gaussian calculations  					###
### > NM data files generated by read_normal_modes.py			###
###																###
### Output:														###
### > 2d_pes_es.dat: 2D PES										###
### > 2d_nm_ampl_es.dat: NM displacement vector surface			###
### > 2d_hess_ampl_es.dat: NM coupling matrix surface			###
### > 2d_hess_diag_es.dat: NM diagonal coupling matrix surface	###
###																###
### > 2d_pes_es.pdf: Contour plot of 2D PES						###			
### > 2d_pes.pdf: Contour plot of 2D PES						###			
###																###
###################################################################

Ha2cm  = 219474.63068 							# Hartree to cm-1 factor
a02ang = .529177210903							# Bohr to Angstrom factor (CODATA 2018)

###################################################################
### define some parameters

### path for NM analysis

path_nm = '../../../NM_READ/'

print('NM path = ',path_nm)

### path for Gaussian log/fchk files

path_log = '../'

print('Logfiles path = ',path_log)

### define tolerances for setting NM to zero

tol_nm = 1.	# for NM displacements
tol_hess = 35.  # for NM hessian [in cm-1]

print('tol_nm = ',tol_nm)
print('tol_hess = ',tol_hess)

### fixed logfiles files

fixed_files=[ [12,1] , [12,4] , [12,5] , [12,8] , [12,9] , [12,10] , [12,12] , [12,13] ]

print('FIXED_FILES = ',fixed_files)

###################################################################
### read grid dimension

inputfile=open(path_log+'2d_scan_relaxed.out','r').readlines()

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

print('Reading energy from file...')

for i in range(ndim1):
	for j in range(ndim2):
		inputfile=path_log+'config_{}_{}.log'.format(i+1,j+1)
		print(inputfile)
		read_energy(i,j,inputfile)

### read energy from '*_fixed' file for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile=path_log+'config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	print(inputfile)
	read_energy(idx,jdx,inputfile)

###################################################################
### mask values with error

#energy_gs = np.ma.array(energy_gs, mask=(energy_gs==energy_gs_error))
#energy_es = np.ma.array(energy_es, mask=(energy_es==energy_es_error))

###################################################################
### shift energy

print('energy_min [au] = ',np.min(energy_gs),np.min(energy_es))
print('energy_max [au] = ',np.max(energy_gs),np.max(energy_es))

energy_0 = np.min(energy_gs)

energy_gs -= energy_0
energy_es -= energy_0

###################################################################
### load NM vectors 
### Note: pos, atmass and nm are in atomic units.

pos0 = np.loadtxt(path_nm+'pos_ts.dat')

atmass = np.loadtxt(path_nm+'atmass_ts.dat')

nm = np.loadtxt(path_nm+'nm_ts.dat')

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

###################################################################
### read positions from Gaussian 16 logfile and compute nm displacement 

nm_ampl = np.zeros([ndim1,ndim2,nfreq])

print('Reading NM displacement from file...')

### loop over Gaussian 16 logfile 

for i in range(ndim1):
	for j in range(ndim2):
		inputfile=path_log+'config_{}_{}.log'.format(i+1,j+1)
		print(inputfile)
		x,y,z = read_pos(inputfile)
		pos1 = mass_weight_pos(x,y,z)
		nm_ampl[i,j,:] = nm_displacement(pos1)

### read nm_ampl from '*_fixed' file for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile=path_log+'config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	print(inputfile)
	x,y,z = read_pos(inputfile)
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

###################################################################
### set to zero NM with small amplitudes
### Note: some NM should be zero (symmetry consideration) but are not 
###       due to numerical error. 

for i in range(nfreq):
	lmax = np.max(np.abs(nm_ampl[:,:,i]))
	if (lmax<tol_nm):
		nm_ampl[:,:,i] = 0.0

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

def project_hess(temp,inputfile):
	''' project scanned NM from hessian '''

	### NM matrix
	nm_aux1 = nm[:,index_x]	
	nm_aux2 = nm[:,index_y]	

	nm_mtrx = np.zeros([ndgree,ndgree])
	for i in range(ndgree):
		for j in range(ndgree):
			nm_mtrx[i,j] += nm_aux1[i]*nm_aux1[j] + nm_aux2[i]*nm_aux2[j]

	### projection matrix
	proj_mtrx = np.identity(ndgree)
	proj_mtrx -= nm_mtrx

	### project hessian

	temp_new = np.matmul(temp,proj_mtrx)  
	temp_new = np.matmul(proj_mtrx,temp_new) 

	return temp_new

###################################################################
### read hessian from Gaussian 16 fchk file 

hess_ampl = np.zeros([ndim1,ndim2,nfreq,nfreq])  

print('Reading hessian from file...')

### loop over Gaussian 16 logfile 

for i in range(ndim1):
	for j in range(ndim2):

		inputfile=path_log+'config_{}_{}.fchk'.format(i+1,j+1)
		print(inputfile)
		hess_temp = read_hess(inputfile)
		hess_temp = mass_weight_hess(hess_temp)
		hess_temp = tridiagonal_to_full_hess(hess_temp)

		inputfile=path_log+'config_{}_{}.log'.format(i+1,j+1)
		hess_temp = project_hess(hess_temp,inputfile)

		hess_temp = cartesian_to_nm_hess(hess_temp)
		hess_ampl[i,j,:,:] = hess_temp

### loop over '*_fixed' files for selected configurations

for i in range(len(fixed_files)):
	idx = fixed_files[i][0] - 1
	jdx = fixed_files[i][1] - 1
	inputfile=path_log+'config_{}_{}_fixed.fchk'.format(idx+1,jdx+1)
	print(inputfile)
	hess_temp = read_hess(inputfile)
	hess_temp = mass_weight_hess(hess_temp)
	hess_temp = tridiagonal_to_full_hess(hess_temp)

	inputfile=path_log+'config_{}_{}_fixed.log'.format(idx+1,jdx+1)
	hess_temp = project_hess(hess_temp,inputfile)

	hess_temp = cartesian_to_nm_hess(hess_temp)
	hess_ampl[idx,jdx,:,:] = hess_temp

###################################################################
### set to zero non-diagonal hessians with small amplitudes

tol_hess = (tol_hess/Ha2cm)**2 	#transform from cm-1 to hessian units 
print('tol_hess = ',tol_hess)

for i in range(nfreq):
	for j in range(nfreq):
		if(i==j):
			continue
		lmax = np.max(np.abs(hess_ampl[:,:,i,j]))
		if (lmax<tol_hess):
			hess_ampl[:,:,i,j] = 0.0

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

	### plot data

	if(lmax0==0): 
		lmax0 = np.max(data0)
	lmin0 = np.min(data0)
	ldel0 = (lmax0-lmin0)/100
	levels0 = np.arange(lmin0,lmax0+ldel0,ldel0)

	if(lmax1==0): 
		lmax1 = np.max(data1)
	lmin1 = np.min(data1)
	ldel1 = (lmax1-lmin1)/100
	levels1 = np.arange(lmin1,lmax1+ldel1,ldel1)

	CS0 = ax[0].contourf(q1,q2,np.transpose(data0),levels=levels0,cmap='jet',extend='max')
	CL0 = ax[0].contour(CS0,levels=CS0.levels[::nlines],colors='k',linewidths=lw)

	CS1 = ax[1].contourf(q1,q2,np.transpose(data1),levels=levels1,cmap='jet',extend='max')
	CL1 = ax[1].contour(CS1,levels=CS1.levels[::nlines],colors='k',linewidths=lw)

	### plot colorbars

	CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax[0])
	CB0.add_lines(CL0)
	CB0.ax.set_title(ctitle,size=labelsz)
	CB0.set_ticks(np.arange(0,levels0[-1]+levels0[1],ldel0*10))

	CS1.cmap.set_over('white')
	CB1 = fig.colorbar(CS1,ax=ax[1])
	CB1.add_lines(CL1)
	CB1.ax.set_title(ctitle,size=labelsz)
	CB1.set_ticks(np.arange(0,levels1[-1]+levels1[1],ldel1*10))

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

	return fig

def plot_2d_1x1(data,title='',xlabel='',ylabel='',ctitle=''):

	### set levels

	lmax = np.max(data)
	lmin = np.min(data)
	ldel = (lmax-lmin)/100
	if(ldel==0.0):
		return
	levels = np.arange(lmin,lmax+ldel,ldel)

	### define figures

	fig,ax= plt.subplots()

	### plot data

	CS0 = ax.contourf(q1,q2,np.transpose(data),levels=levels,cmap='jet',extend='max')
	CL0 = ax.contour(CS0,levels=CS0.levels[::nlines],colors='k',linewidths=lw)

	### plot colorbars

	CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax)
	CB0.add_lines(CL0)
	CB0.ax.set_title(ctitle,size=labelsz)

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

	return fig 

###################################################################
### plot data

xlabel = xlabel0 = xlabel1 = r'$\mathregular{Q_{1}}$ / a.u. '
ylabel = ylabel0 = ylabel1 = r'$\mathregular{Q_{5}}$ / a.u. '

#---------------------------------------
### PES
data0 = energy_es * Ha2cm
data1 = energy_gs * Ha2cm
title0 = 'Excited state'
title1 = 'Ground state'
ctitle = r'$\mathregular{V \ / \ cm^{-1}}$'

fig1 = plot_2d_2x1(data0,data1,title0=title0,title1=title1,xlabel1=xlabel1,ylabel0=ylabel0,ylabel1=ylabel1,ctitle=ctitle,lmax1=6000)

fig2 = plot_2d_1x1(data0,title=title0,xlabel=xlabel1,ylabel=ylabel0,ctitle=ctitle)


#---------------------------------------
### nm_ampl 

### index of NM to plot
idx = []
#idx = [9,11,61,44]

for i in idx:
	data = nm_ampl[:,:,i]
	title = 'NM amplitude # {}'.format(i+1)
	ctitle = r'$\mathregular{Q_0 \ / \ a.u.}$'
	plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

#---------------------------------------
### hess_ampl 

### index of NM to plot
idx = []
#idx = [ [11,11], [11,9], [2,68] ]

for i in idx:
	data = hess_ampl[:,:,i[0],i[1]]
	fct = np.sign(data)
	data = fct*np.power(fct*data,0.5) * Ha2cm
	title = 'hess amplitude # {} {}'.format(i[0]+1,i[1]+1)
	ctitle = r'$\mathregular{H \ / \ cm^{-1}}$'
	plot_2d_1x1(data,title=title,xlabel=xlabel,ylabel=ylabel,ctitle=ctitle)

###################################################################
### show plot

plt.show()

###################################################################
### save plot

fig1.savefig('2d_pes.pdf',format='pdf',dpi=1200)

fig2.savefig('2d_pes_es.pdf',format='pdf',dpi=1200)

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


