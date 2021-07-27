#!/usr/bin/python

###################################################################
### Read 2D data from scan and interpolate/extrapolate		###
###								### 
### Note: data read in atomic units				###
###################################################################

from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt
import sys

###################################################################
### define some parameters 

ndim1 = 2**5		# dimensions of new grid along x
ndim2 = 2**5		# dimensions of new grid along y

q1_max = 100.		# maximum range of new grid along x
q1_min = -100.		# minimum range of new grid along x

q2_max = 250.		# maximum range of new grid along y
q2_min = -250.		# minimum range of new grid along y

#pot_type='gs'		# Ground State potential
pot_type='es'		# Excited State potential

#interp_kind='linear'	# kind of interpolation
interp_kind='cubic'	# kind of interpolation

###################################################################
### define input/output files

if(pot_type=='gs'):
	potfile_in    = '2d_pes_gs.dat'
	potfile_out   = '2d_pes_gs_interpolated.dat'
	nmfile_in     = '2d_nm_ampl_gs.dat'
	nmfile_out    = '2d_nm_ampl_gs_interpolated.dat'
	diagfile_in   = '2d_hess_diag_gs.dat'
	diagfile_out  = '2d_hess_diag_gs_interpolated.dat'
	hessfile_in   = '2d_hess_ampl_gs.dat'
	hessfile_out  = '2d_hess_ampl_gs_interpolated.dat'
elif(pot_type=='es'):
	potfile_in    = '2d_pes_es.dat'
	potfile_out   = '2d_pes_es_interpolated.dat'
	nmfile_in     = '2d_nm_ampl_es.dat'
	nmfile_out    = '2d_nm_ampl_es_interpolated.dat'
	diagfile_in   = '2d_hess_diag_es.dat'
	diagfile_out  = '2d_hess_diag_es_interpolated.dat'
	hessfile_in   = '2d_hess_ampl_es.dat'
	hessfile_out  = '2d_hess_ampl_es_interpolated.dat'
else:
	sys.exit('Error. pot_type should be "gs" or "es" \n')

###################################################################
### define interpolation/extrapolation function
### Note: Taken from https://github.com/pig2015/mathpy/blob/master/polation/globalspline.py

class GlobalSpline2D(interpolate.interp2d):
    def __init__(self, x, y, z, kind='linear'):
        if kind == 'linear':
            if len(x) < 2 or len(y) < 2:
                raise self.get_size_error(2, kind)
        elif kind == 'cubic':
            if len(x) < 4 or len(y) < 4:
                raise self.get_size_error(4, kind)
        elif kind == 'quintic':
            if len(x) < 6 or len(y) < 6:
                raise self.get_size_error(6, kind)
        else:
            raise ValueError('unidentifiable kind of spline')

        super(GlobalSpline2D,self).__init__(x, y, z, kind=kind)
        self.extrap_fd_based_xs = self._linspace_10(self.x_min, self.x_max, -4)
        self.extrap_bd_based_xs = self._linspace_10(self.x_min, self.x_max, 4)
        self.extrap_fd_based_ys = self._linspace_10(self.y_min, self.y_max, -4)
        self.extrap_bd_based_ys = self._linspace_10(self.y_min, self.y_max, 4)

    @staticmethod
    def get_size_error(size, spline_kind):
        return ValueError('length of x and y must be larger or at least equal '
                          'to {} when applying {} spline, assign arrays with '
                          'length no less than '
                          '{}'.format(size, spline_kind, size))

    @staticmethod
    def _extrap1d(xs, ys, tar_x):
        if isinstance(xs, np.ndarray):
            xs = np.ndarray.flatten(xs)
        if isinstance(ys, np.ndarray):
            ys = np.ndarray.flatten(ys)
        assert len(xs) >= 4
        assert len(xs) == len(ys)
        f = interpolate.InterpolatedUnivariateSpline(xs, ys)
        return f(tar_x)

    @staticmethod
    def _linspace_10(p1, p2, cut=None):
        ls = list(np.linspace(p1, p2, 10))
        if cut is None:
            return ls
        assert cut <= 10
        return ls[-cut:] if cut < 0 else ls[:cut]

    def _get_extrap_based_points(self, axis, extrap_p):
        if axis == 'x':
            return (self.extrap_fd_based_xs if extrap_p > self.x_max else
                    self.extrap_bd_based_xs if extrap_p < self.x_min else [])
        elif axis == 'y':
            return (self.extrap_fd_based_ys if extrap_p > self.y_max else
                    self.extrap_bd_based_ys if extrap_p < self.y_min else [])
        assert False, 'axis unknown'
        
    def __call__(self, x_, y_, **kwargs):
        xs = np.atleast_1d(x_)
        ys = np.atleast_1d(y_)

        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")

        pz_yqueue = []
        for y in ys:
            extrap_based_ys = self._get_extrap_based_points('y', y)

            pz_xqueue = []
            for x in xs:
                extrap_based_xs = self._get_extrap_based_points('x', x)

                if not extrap_based_xs and not extrap_based_ys:
                    # inbounds
                    pz = super(GlobalSpline2D,self).__call__(x, y, **kwargs)[0]

                elif extrap_based_xs and extrap_based_ys:
                    # both x, y atr outbounds
                    # allocate based_z from x, based_ys
                    extrap_based_zs = self.__call__(x,
                                                    extrap_based_ys,
                                                    **kwargs)
                    # allocate z of x, y from based_ys, based_zs
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y)

                elif extrap_based_xs:
                    # only x outbounds
                    extrap_based_zs = super(GlobalSpline2D,self).__call__(extrap_based_xs,
                                                       y,
                                                       **kwargs)
                    pz = self._extrap1d(extrap_based_xs, extrap_based_zs, x)

                else:
                    # only y outbounds
                    extrap_based_zs = super(GlobalSpline2D,self).__call__(x,
                                                       extrap_based_ys,
                                                       **kwargs)
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y)

                pz_xqueue.append(pz)

            pz_yqueue.append(pz_xqueue)

        zss = pz_yqueue
        if len(zss) == 1:
            zss = zss[0]
        return np.array(zss)

###################################################################
### define plot functions

nlines = 5

def plot_2d(data0,data1,title=''):

	### check if there is data to plot
	if(np.min(data0)==np.max(data0) or np.min(data1)==np.max(data1)):
		return

	### define figure

	fig,ax= plt.subplots(2,sharex=True,sharey=True)
	#fig.subplots_adjust(bottom=0.12,right=0.95,top=0.92)

	### define levels

	#levels0 = np.linspace(np.min(data0),np.max(data0),100)
	#levels1 = np.linspace(np.min(data1),np.max(data1),100)
	lmin = min(np.min(data0),np.min(data1))
	lmax = max(np.max(data0),np.max(data1))
	levels0 = np.linspace(lmin,lmax,100)
	levels1 = np.linspace(lmin,lmax,100)

	### plot data

	CS0 = ax[0].contourf(q1_old,q2_old,np.transpose(data0),levels=levels0,cmap='jet')#,extend='both')
	CL0 = ax[0].contour(CS0,levels=CS0.levels[::nlines],colors='k',linewidths=1.)

	CS1 = ax[1].contourf(q1,q2,np.transpose(data1),levels=levels1,cmap='jet')#,extend='both')
	CL1 = ax[1].contour(CS1,levels=CS1.levels[::nlines],colors='k',linewidths=1.)

	### plot colorbars

	##CS0.cmap.set_under('yellow')
	#CS0.cmap.set_over('white')
	CB0 = fig.colorbar(CS0,ax=ax)
	CB0.add_lines(CL0)
#	CB0.ax.set_title(ctitle,size=labelsz)
#	CB0.set_ticks(np.arange(0,levels0[-1]+levels0[1],ldel0*10))

	### set labels

	fig.suptitle(title)

	ax[0].set_title('Raw')
	ax[1].set_title('Interpolated')

	ax[1].set_xlabel(r'$\mathregular{x \ / \ au}$')
	ax[0].set_ylabel(r'$\mathregular{y \ / \ au}$')
	ax[1].set_ylabel(r'$\mathregular{y \ / \ au}$')

	return fig

###################################################################
### define grid coordinates

q1 = np.linspace(q1_min,q1_max,ndim1,endpoint=True) 
q2 = np.linspace(q2_min,q2_max,ndim2,endpoint=True) 

print('ndim = ',ndim1,ndim2)
print('q_max [au] = ',q1[-1],q2[-1]) 
print('q_min [au] = ',q1[0],q2[0]) 

###################################################################
### read potential energy and 2D coordinate
### Note: V and Q read in atomic units

inputfile=open(potfile_in,'r').readlines()

### get dimension of q1 and q2 coordinates
ndim1_old = int(inputfile[0].split('npoints =')[1].split()[0])
ndim2_old = int(inputfile[0].split('npoints =')[1].split()[1])
print('ndim_old = ',ndim1_old,ndim2_old)

### load data
data=np.loadtxt(inputfile)

### get q1 and q2 coordinates from first two columns
q1_old = data[0::ndim2_old,0] 
q2_old = data[0:ndim2_old,1] 

### remove two first columns (q1 and q2 coordinates)
data = np.delete(data,0,axis=1)
data = np.delete(data,0,axis=1)

### reshape data
V_old = np.reshape(data,[ndim1_old,ndim2_old])

###################################################################
### interpolate/extrapolate the potential
### Note: order coordinates as (y,x) to keep V[i,j] as [x,y] array

print('Interpolating potential...')

f = GlobalSpline2D(q2_old,q1_old,V_old,kind=interp_kind)
V = f(q2,q1)

print('Interpolating potential...done')

### shift energy

print('energy_min [au] = ',np.min(V))
print('energy_max [au] = ',np.max(V))

#V_0 = np.min(V)
#V -= V_0

###################################################################
### plot interpolated potential

title = 'Interpolation GS'
fig1 = plot_2d(V_old,V,title=title)

#plt.show()

#fig1.savefig('2d_pes_gs_interpolated.pdf',format='pdf',dpi=1200)

###################################################################
### save interpolated potential

print('Saving Interpolated potential...')

ofile=open(potfile_out,'w')

### print header 
ofile.write('# x (au) y (au) V (au) ')
ofile.write('npoints = {} {} q1_range = {} {} q2_range = {} {} \n'.format(ndim1,ndim2,q1_min,q1_max,q2_min,q2_max))

### save data
for i in range(ndim1):
	for j in range(ndim2):
		ofile.write('{} {} {} \n'.format(q1[i],q2[j],V[i,j]))

ofile.close()

###################################################################
### read nm amplitudes
### Note: NM read in atomic units

inputfile=open(nmfile_in,'r').readlines()

### get number of NM
nmodes = int(inputfile[0].split('nmodes =')[1].split()[0])
print('nmodes = ',nmodes)

index_x = int(inputfile[0].split('index =')[1].split()[0])
index_y = int(inputfile[0].split('index =')[1].split()[1])
print('index_x/y = ',index_x,index_y)

### load data
data=np.loadtxt(inputfile)

### remove two first columns (q1 and q2 coordinates)
data = np.delete(data,0,axis=1)
data = np.delete(data,0,axis=1)

### reshape data
nm_ampl_old = np.reshape(data,[ndim1_old,ndim2_old,nmodes])

###################################################################
### interpolate/extrapolate the NM amplitude
### Note: order coordinates as (y,x) to keep nm_ampl[i,j] as [x,y] array

nm_ampl = np.zeros([ndim1,ndim2,nmodes])

for k in range(nmodes):
	print('Interpolating nm ampl # {} of {}...'.format(k,nmodes))
	lmin = np.min(nm_ampl_old[:,:,k])
	lmax = np.max(nm_ampl_old[:,:,k])
	if(not (lmin==0 and lmax==0)):  # skipping modes with zero amplitude 
		f = GlobalSpline2D(q2_old,q1_old,nm_ampl_old[:,:,k],kind=interp_kind)
		nm_ampl[:,:,k] = f(q2,q1)

###################################################################
### plot interpolation NM amplitudes

for k in range(10):#nmodes):
	title = 'Interpolation NM amplitude # {}'.format(k+1)
	plot_2d(nm_ampl_old[:,:,k],nm_ampl[:,:,k],title=title)

#idx = 9-1-2
#title = 'Interpolation NM amplitude # {}'.format(idx+1)
#fig3 = plot_2d(nm_ampl_old[:,:,idx],nm_ampl[:,:,idx],title=title)

#idx = 11-1-2
#title = 'Interpolation NM amplitude # {}'.format(idx+1)
#fig4 = plot_2d(nm_ampl_old[:,:,idx],nm_ampl[:,:,idx],title=title)

#plt.show()

###################################################################
### save interpolated NM amplitudes

print('Saving Interpolated NM amplitudes...')

ofile=open(nmfile_out,'w')

### print header 
ofile.write('# x (au) y (au) [Q0_i] ')
ofile.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1_min,q1_max,q2_min,q2_max))
ofile.write('nmodes = {} index = {} {} \n'.format(nmodes,index_x,index_y))

### save data
for i in range(ndim1):
	for j in range(ndim2):
		ofile.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nmodes):
			ofile.write('{} '.format(nm_ampl[i,j,k]))
		ofile.write('\n')
ofile.close()

###################################################################
### read diagonal nm couplings
### Note: NM couplings read in atomic units

inputfile=open(diagfile_in,'r').readlines()

### load data
data=np.loadtxt(inputfile)

### remove two first columns (q1 and q2 coordinates)
data = np.delete(data,0,axis=1)
data = np.delete(data,0,axis=1)

### reshape data
hess_diag_old = np.reshape(data,[ndim1_old,ndim2_old,nmodes])

###################################################################
### interpolate/extrapolate the diagonal NM couplings
### Note: order coordinates as (y,x) to keep hess_ampl[i,j] as [x,y] array

hess_diag = np.zeros([ndim1,ndim2,nmodes])

for k in range(nmodes):
	print('Interpolating diagonal nm hess # {} of {}...'.format(k,nmodes))
	f = GlobalSpline2D(q2_old,q1_old,hess_diag_old[:,:,k],kind=interp_kind)
	hess_diag[:,:,k] = f(q2,q1)

###################################################################
### plot interpolation diagonal NM couplings

#for k in range(nmodes):
#	title = 'Interpolation diag NM hess # {}'.format(k+1)
#	plot_2d(hess_diag_old[:,:,k],hess_diag[:,:,k],title=title)

idx = 9-1-2
title = 'Interpolation diag NM hess # {}'.format(idx+1)
fig5 = plot_2d(hess_diag_old[:,:,idx],hess_diag[:,:,idx],title=title)

idx = 11-1-2
title = 'Interpolation diag NM hess # {}'.format(idx+1)
fig6 = plot_2d(hess_diag_old[:,:,idx],hess_diag[:,:,idx],title=title)

#plt.show()

###################################################################
### save interpolated NM diagonal hessian

print('Saving Interpolated NM diagonal hessian...')

ofile= open(diagfile_out,'w')

### print header 
ofile.write('# x (au) y (au) [H_ii] ')
ofile.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1_min,q1_max,q2_min,q2_max))
ofile.write('nmodes = {} index = {} {} \n'.format(nmodes,index_x,index_y))

### save data
for i in range(ndim1):
	for j in range(ndim2):
		ofile.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nmodes):
			ofile.write('{} '.format(hess_diag[i,j,k]))
		ofile.write('\n')
ofile.close()

###################################################################
### read nm couplings
### Note: NM couplings read in atomic units
### Note: NM couplings read in tridiagonal form

inputfile=open(hessfile_in,'r').readlines()

### load data
data=np.loadtxt(inputfile)

### remove two first columns (q1 and q2 coordinates)
data = np.delete(data,0,axis=1)
data = np.delete(data,0,axis=1)

### reshape data
nhess=len(np.transpose(data))
#hess_ampl_old = np.reshape(data,[ndim1_old,ndim2_old,nmodes,nmodes])
hess_ampl_old = np.reshape(data,[ndim1_old,ndim2_old,nhess])

###################################################################
### interpolate/extrapolate the NM couplings
### Note: order coordinates as (y,x) to keep hess_ampl[i,j] as [x,y] array

'''
hess_ampl = np.zeros([ndim1,ndim2,nmodes,nmodes])

for k in range(nmodes):
	for l in range(nmodes):
		print('Interpolating nm hess # {} {}...'.format(k,l))
		lmin = np.min(hess_ampl_old[:,:,k,l])
		lmax = np.max(hess_ampl_old[:,:,k,l])
		if(not (lmin==0 and lmax==0)):  # skipping modes with zero amplitude 
			f = GlobalSpline2D(q2_old,q1_old,hess_ampl_old[:,:,k,l],kind=interp_kind)
			hess_ampl[:,:,k,l] = f(q2,q1)
'''

hess_ampl = np.zeros([ndim1,ndim2,nhess])

for k in range(nhess):
	print('Interpolating nm hess # {} of {}...'.format(k,nhess))
	lmin = np.min(hess_ampl_old[:,:,k])
	lmax = np.max(hess_ampl_old[:,:,k])
	if(not (lmin==0 and lmax==0)):  # skipping modes with zero amplitude 
		f = GlobalSpline2D(q2_old,q1_old,hess_ampl_old[:,:,k],kind=interp_kind)
		hess_ampl[:,:,k] = f(q2,q1)

###################################################################
### plot interpolation NM couplings
'''
idx = 11-1-2
jdx = 11-1-2
title = 'Interpolation NM coupling # {} {}'.format(idx+1,jdx+1)
fig7 = plot_2d(hess_ampl_old[:,:,idx,jdx],hess_ampl[:,:,idx,jdx],title=title)

idx = 9-1-2
jdx = 11-1-2
title = 'Interpolation NM coupling # {} {}'.format(idx+1,jdx+1)
fig8 = plot_2d(hess_ampl_old[:,:,idx,jdx],hess_ampl[:,:,idx,jdx],title=title)
'''

idx = 8
title = 'Interpolation NM coupling # {}'.format(idx+1)
fig7 = plot_2d(hess_ampl_old[:,:,idx],hess_ampl[:,:,idx],title=title)

idx = 9
title = 'Interpolation NM coupling # {}'.format(idx+1)
fig8 = plot_2d(hess_ampl_old[:,:,idx],hess_ampl[:,:,idx],title=title)


#plt.show()

###################################################################
### save interpolated hessian amplitude

print('Saving Interpolated hessian amplitude...')

ofile=open(hessfile_out,'w')

### print header 
ofile.write('# x (au) y (au) [H_ij] ')
ofile.write('npoints = {} {} q1_range = {} {} q2_range = {} {} '.format(ndim1,ndim2,q1_min,q1_max,q2_min,q2_max))
ofile.write('nmodes = {} index = {} {} \n'.format(nmodes,index_x,index_y))

for i in range(ndim1):
	for j in range(ndim2):
		ofile.write('{} {} '.format(q1[i],q2[j]))
		for k in range(nhess):
			ofile.write('{} '.format(hess_ampl[i,j,k]))
		ofile.write('\n')
ofile.close()

###################################################################
### show plots

plt.show()

###################################################################
### end program

sys.exit(0)


