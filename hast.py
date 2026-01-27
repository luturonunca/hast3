import gc
import os
import re
import sys
import glob
import math
import copy
import pynbody
import warnings
import matplotlib
import numpy as np
import configparser
import seaborn as sns
import astropy.units as u
import numpy.linalg as la
import matplotlib.pyplot as pyplot
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from numpy.polynomial.polynomial import polyfit
from astropy.cosmology import Planck15, z_at_value

warnings.filterwarnings("ignore")

def _load_sim(path):
    sim = pynbody.load(path)
    try:
        sim.physical_units()
    except Exception:
        pass
    return sim

mpl_major = int(matplotlib.__version__[0])
mpl_minor = int(matplotlib.__version__[2])
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cp = sns.color_palette(flatui)

if ( mpl_major >= 2 or (mpl_major==1 and mpl_minor>=5) ):
    mpl_colormap = 'plasma'
else:
    mpl_colormap = 'gist_heat'

def __version():
      print('     ___           ___           ___                  ')
      print('    /\  \         /\  \         /\__\                 ')
      print('    \:\  \       /::\  \       /:/ _/_         ___    ')
      print('     \:\  \     /:/\:\  \     /:/ /\  \       /\__\   ')
      print(' ___ /::\  \   /:/ /::\  \   /:/ /::\  \     /:/  /   ')
      print('/\  /:/\:\__\ /:/_/:/\:\__\ /:/_/:/\:\__\   /:/__/    ')
      print('\:\/:/  \/__/ \:\/:/  \/__/ \:\/:/ /:/  /  /::\  \    ')
      print(' \::/__/       \::/__/       \::/ /:/  /  /:/\:\  \   ')
      print('  \:\  \        \:\  \        \/_/:/  /   \/__\:\  \  ')
      print('   \:\__\        \:\__\         /:/  /         \:\__\ ')
      print('    \/__/         \/__/         \/__/           \/__/ ')
      print('| ------------------------------------------------------------')
      print('| HAlo Selection Tools - Version 0.5')

def __unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

class config_selection_obj():
    def parse_input(self, ConfigFile):
        config = configparser.SafeConfigParser({'fname':'music_zoom','recompute_rtb':False,'plot':False})
        config.read(ConfigFile)

        self.output_zinit = config.get('selection','output_zinit')
        self.output_zlast = config.get('selection','output_zlast')
        self.min_mass = config.getfloat('selection','min_mass')
        self.max_mass = config.getfloat('selection','max_mass')
        self.max_mass_neighb = config.getfloat('selection','max_mass_neighb')
        self.rtb = config.getfloat('selection','rtb')
        self.rbuffer = config.getfloat('selection','rbuffer')
        try:
            self.xsearch = config.getfloat('selection','xsearch')
            self.ysearch = config.getfloat('selection','ysearch')
            self.zsearch = config.getfloat('selection','zsearch')
            self.rsearch = config.getfloat('selection','rsearch')
        except:
            self.xsearch = 0.5
            self.ysearch = 0.5
            self.zsearch = 0.5
            self.rsearch = -1.0
        try:
            self.min_neighb = config.getint('selection','min_neighb')
        except:
            self.min_neighb = 0
        try:
            self.max_neighb = config.getint('selection','max_neighb')
        except:
            self.max_neighb = 100000
        self.fname = config.get('selection','fname')
        try:
            self.plot = config.getboolean('selection','plot')
        except:
            self.plot = True
        try:
            self.plot_traceback = config.getboolean('selection','plot_traceback')
        except:
            self.plot_traceback = False
        try:
            self.tree_nleaves = config.getint('selection','tree_nleaves')
        except:
            self.tree_nleaves = 100
        try:
            self.boundary_min = config.getfloat('selection','boundary_min')
        except:
            self.boundary_min = 0.1
        try:
            self.boundary_max = config.getfloat('selection','boundary_max')
        except:
            self.boundary_max = 0.9
        try:
            self.clump_mass_unit = config.get('selection','clump_mass_unit')
        except:
            self.clump_mass_unit = 'fraction'
        try:
            self.halo_finder = config.get('selection','halo_finder')
        except:
            self.halo_finder = 'ramses'

class config_decontamination_obj():
    def parse_input(self, ConfigFile):
        config = configparser.SafeConfigParser()
        config.read(ConfigFile)


        self.output_zinit = config.get('decontamination','output_zinit')
        self.output_zlast = config.get('decontamination','output_zlast')
        self.rbuffer = config.getfloat('decontamination','rbuffer')
        try:
            self.rexclude = config.getfloat('decontamination','rexclude')
        except:
            self.rexclude = 10.

        self.output_dir = config.get('decontamination','output_dir')
        try:
            self.rvir = config.getfloat('decontamination','rvir')
        except:
            self.rvir = 1.0
        try:
            self.rvir_track = config.getfloat('decontamination','rvir_track')
        except:
            self.rvir_track = 0.25
        try:
            self.rvir_search = config.getfloat('decontamination','rvir_search')
        except:
            self.rvir_search = 5.0
        try:
            self.aexp_min = config.getfloat('decontamination','aexp_min')
        except:
            self.aexp_min = 0.0
        self.fname = config.get('decontamination','fname')
        try:
            halo_coords_str = config.get('decontamination','halo_coords')
            self.halo_coords = (np.array(re.split(',|;',''.join(halo_coords_str.split())))).astype(np.float)
        except:
            self.halo_coords = np.array([-1.0,-1.0,-1.0])
        try:
            self.halo_num = config.getint('decontamination','halo_num')
        except:
            self.halo_num = 1
        try:
            self.plot = config.getboolean('decontamination','plot')
        except:
            self.plot = True
        try:
            self.tree_nleaves = config.getint('decontamination','tree_nleaves')
        except:
            self.tree_nleaves = 100
        try:
            ps_str = config.get('decontamination','point_shift')
            self.ps = (np.array(re.split(',|;',''.join(ps_str.split())))).astype(np.int)
        except:
            self.ps = np.array([0,0,0]).astype(np.int)
        try:
            self.pslmin = config.getint('decontamination','point_shift_lmin')
        except:
            self.pslmin = 1
        try:
            self.halo_cutoff = config.getfloat('decontamination','halo_cutoff')
        except:
            self.halo_cutoff = 1e3
        try:
            self.halo_massfrac = config.getfloat('decontamination','halo_massfrac')
        except:
            self.halo_massfrac = 0.10
        try:
            self.rank_function = config.get('decontamination','rank_function')
        except:
            self.rank_function = 'mass'


class config_tracking_obj():
    def parse_input(self, ConfigFile):
        config = configparser.SafeConfigParser()
        config.read(ConfigFile)

        self.output_dir = config.get('tracking','output_dir')
        try:
            self.rvir = config.getfloat('tracking','rvir')
        except:
            self.rvir = 0.25
        try:
            self.rvir_search = config.getfloat('tracking','rvir_search')
        except:
            self.rvir_search = 5.0
        try:
            self.aexp_min = config.getfloat('tracking','aexp_min')
        except:
            self.aexp_min = 0.0
        self.fname = config.get('tracking','fname')
        try:
            halo_coords_str = config.get('tracking','halo_coords')
            self.halo_coords = (np.array(re.split(',|;',''.join(halo_coords_str.split())))).astype(np.float)
        except:
            self.halo_coords = np.array([-1.0,-1.0,-1.0])
        try:
            self.plot = config.getboolean('tracking','plot')
        except:
            self.plot = True
        try:
            self.tree_nleaves = config.getint('tracking','tree_nleaves')
        except:
            self.tree_nleaves = 100
        try:
            self.halo_cutoff = config.getfloat('tracking','halo_cutoff')
        except:
            self.halo_cutoff = 1e3
        try:
            self.halo_massfrac = config.getfloat('tracking','halo_massfrac')
        except:
            self.halo_massfrac = 0.10
        try:
            self.rank_function = config.get('tracking','rank_function')
        except:
            self.rank_function = 'mass'


class config_analysis_obj():
    def parse_input(self, ConfigFile):
        config = configparser.SafeConfigParser()
        config.read(ConfigFile)

        self.output = config.get('analysis','output')
        self.rvr = config.getfloat('analysis','rvr')
        self.rbuffer = config.getfloat('analysis','rbuffer')
        try:
            self.halo_num = config.getfloat('analysis','halo_num')
        except:
            self.halo_num = 1
        try:
            self.nbin_sfr = config.getint('analysis','nbin_sfr')
        except:
            self.nbin_sfr = 100
        try:
            self.tree_nleaves = config.getint('analysis','tree_nleaves')
        except:
            self.tree_nleaves = 100
        try:
            self.plot = config.getboolean('analysis','plot')
        except:
            self.plot = True
        try:
            self.rank_function = config.get('analysis','rank_function')
        except:
            self.rank_function = 'mass'


def halo_list(output,quiet=False,clump_mass_unit='fraction'):

    list = glob.glob(output+'/clump_?????.txt?????')
    if(not quiet):
        print('| ------------------------------------------------------------')
        print('| Reading RAMSES clump finder files')
        print('| ------------------------------------------------------------')
        print('| nfiles        = {0}'.format(len(list)))
    i=0
    for file in list:
        data = np.loadtxt(file,skiprows=1,dtype=None)
        if(np.size(data)==0):
            continue
        if(i>0):
            data_all = np.vstack((data_all,data))
        else:
            data_all = data
        i=i+1
    data_sorted = data_all[data_all[:,10].argsort()]
    d = _load_sim(output)
    # Convert clump positions from code units (0..1) to kpc when needed.
    try:
        boxsize_kpc = float(d.properties['boxsize'].in_units('kpc'))
    except Exception:
        boxsize_kpc = None
    if boxsize_kpc is not None:
        if np.max(data_sorted[:,4:7]) <= 1.0:
            data_sorted[:,4:7] *= boxsize_kpc
    mass = d.d['mass']
    if hasattr(mass, "in_units"):
        mass_msol = mass.in_units("Msol")
    else:
        # Fallback for arrays without unit support (assume already in Msol).
        mass_msol = mass
    total_mass = float(np.sum(mass_msol))
    particle_mass = float(np.min(mass_msol))
    if clump_mass_unit == 'fraction':
        data_sorted[:,10] *= total_mass
    elif clump_mass_unit in ('particle', 'particles'):
        data_sorted[:,10] *= particle_mass
    elif clump_mass_unit == 'msol':
        pass
    else:
        if not quiet:
            print('[Warning] Unknown clump_mass_unit={0}; using fraction'.format(clump_mass_unit))
        data_sorted[:,10] *= total_mass
    if(not quiet):
        min = np.min(data_sorted[:,10])
        max = np.max(data_sorted[:,10])
        min_part_mass = float(np.min(mass_msol))
        max_part_mass = float(np.max(mass_msol))
        print('| Min mass      = {0:.2e} Msol'.format(min))
        print('| Max mass      = {0:.2e} Msol'.format(max))
        print('| Min part mass = {0:.3e} Msol'.format(min_part_mass))
        print('| Max part mass = {0:.3e} Msol'.format(max_part_mass))
        print('| Total mass    = {0:.2e} Msol'.format(total_mass))
        print('| ------------------------------------------------------------')
    return data_sorted


def _halo_center_to_unit_box(sim, halo):
    pos = None
    if hasattr(halo, "properties"):
        props = halo.properties
        if "pos" in props:
            pos = props["pos"]
        elif all(k in props for k in ("Xc", "Yc", "Zc")):
            pos = np.array([props["Xc"], props["Yc"], props["Zc"]])
    if pos is None:
        pos = np.mean(halo["pos"], axis=0)

    pos_arr = np.array(pos)
    if np.all(pos_arr >= 0.0) and np.all(pos_arr <= 1.0):
        return pos_arr

    boxsize = sim.properties.get("boxsize", None)
    if boxsize is None:
        return pos_arr

    try:
        if hasattr(pos, "in_units") and hasattr(boxsize, "in_units"):
            pos_units = str(pos.units)
            pos_val = pos.in_units(pos_units)
            box_val = boxsize.in_units(pos_units)
            return np.array(pos_val) / float(box_val)
    except Exception:
        pass

    try:
        return pos_arr / float(boxsize)
    except Exception:
        return pos_arr


def _halo_mass_msol(halo):
    mass = None
    if hasattr(halo, "properties") and "mass" in halo.properties:
        mass = halo.properties["mass"]
    if mass is None:
        mass = np.sum(halo["mass"])
    if hasattr(mass, "in_units"):
        return float(mass.in_units("Msol"))
    return float(mass)


def halo_list_pynbody(sim, halo_finder, quiet=False):
    if not quiet:
        print('| ------------------------------------------------------------')
        print('| Reading halo catalogue via pynbody')
        print('| ------------------------------------------------------------')

    try:
        try:
            halos = sim.halos(halo_finder)
        except TypeError:
            halos = sim.halos(halo_finder=halo_finder)
    except Exception:
        print('[Error] halo finder "{0}" not available'.format(halo_finder))
        sys.exit()

    n_halos = len(halos)
    if not quiet:
        print('| nhalos        = {0}'.format(n_halos))

    data = np.zeros((n_halos, 11), dtype=float)
    for i in range(n_halos):
        data[i, 0] = i
        data[i, 4:7] = _halo_center_to_unit_box(sim, halos[i])
        data[i, 10] = _halo_mass_msol(halos[i])

    data_sorted = data[data[:, 10].argsort()]
    return data_sorted

def __halo_list_tracking(output,conf):

    list = glob.glob(output+'/clump_?????.txt?????')
    i=0
    for file in list:
                data = np.loadtxt(file,skiprows=1,dtype=None)
                if(np.size(data)==0):
                        continue
                if(i>0):
                        data_all = np.vstack((data_all,data))
                else:
                        data_all = data
                i=i+1
    if(conf.rank_function == 'mass'):
            c = data_all[:,10]
    elif(conf.rank_function == 'ncell'):
            c = data_all[:,3]
    elif(conf.rank_function == 'rho_max'):
            c = data_all[:,8]
    elif(conf.rank_function == 'rho_ave'):
            c = data_all[:,9]
    elif(conf.rank_function == 'mass_rho'):
            c = (1e4*data_all[:,3]/np.max(data_all[:,3]))*(data_all[:,8]/np.max(data_all[:,8]))
    else:
            c = data_all[:,10]
    sorted = np.argsort(c)
    data_sorted = data_all[sorted]
    data_sorted = data_sorted[::-1]
    d = _load_sim(output)
    # Convert clump positions from code units (0..1) to kpc when needed.
    try:
        boxsize_kpc = float(d.properties['boxsize'].in_units('kpc'))
    except Exception:
        boxsize_kpc = None
    if boxsize_kpc is not None:
        if np.max(data_sorted[:,4:7]) <= 1.0:
            data_sorted[:,4:7] *= boxsize_kpc
    return data_sorted



def plot_candidates(data,sim,center=[0.,0.,0.]):
    sns.set_context('poster')
    sns.set_style("ticks",{"axes.grid": False,"xtick.direction":'in',"ytick.direction":'in'})
    cp2 = sns.color_palette("Set1",len(data[:,0]))
    print('| Plotting ',len(data[:,0]),' haloes')
    fig,ax = pyplot.subplots(1,2,figsize=(18,8),sharex=True)
    proj =[['y','x'],['z','x']]
    dproj =[[5,4],[6,4]]
    for i in range(len(ax)):
        x=proj[i][0]
        y=proj[i][1]
        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)
        if np.max(sim.d[x]) > 1.0 or np.max(sim.d[y]) > 1.0:
            try:
                boxsize = float(sim.properties['boxsize'].in_units('kpc'))
            except Exception:
                boxsize = float(np.max([np.max(sim.d[x]), np.max(sim.d[y])]))
            hist_range = [[0.0, boxsize], [0.0, boxsize]]
        else:
            hist_range = [[0.0, 1.0], [0.0, 1.0]]
        im,xedges,yedges = np.histogram2d(
            sim.d[x], sim.d[y], weights=sim.d['mass'], bins=512, range=hist_range)
        im = np.rot90(im)
        b = ax[i].get_position()
        data[:,4:7] -= center
        h = ax[i].scatter(data[:,dproj[i][0]],data[:,dproj[i][1]],s=50,c=cp2,alpha=0.5)
        ax[i].set(adjustable='box-forced', aspect='equal')
        extent_max = hist_range[0][1]
        tv = ax[i].imshow(
            np.log10(im), cmap='bone_r', interpolation='quadric',
            aspect='equal', extent=[0.0, extent_max, 0.0, extent_max])
        ax[i].set_xlim([0.0-center[0], extent_max-center[0]])
        ax[i].set_ylim([0.0-center[1], extent_max-center[1]])
        for j in range(len(data[:,0])):
            ax[i].annotate(str(j+1),(data[j,dproj[i][0]]+0.01,data[j,dproj[i][1]]+0.01),color=cp2[j])

    return ax

def find_region(data,radius,nregion):
    x = np.squeeze(data[:,4:7])
    print('| Building Tree with {0} haloes'.format(len(data[:,0])))
    tree = KDTree(x)
    np.random.seed(0)
    print('| Querying halo Tree')
    rp = np.random.random((nregion, 3))
    res = tree.query_radius(rp,radius)
    return rp,res

def find_galaxy(data,radius,min_mass,max_mass):
    x = np.squeeze(data[:,4:7])
    print('| Building Tree with {0} haloes'.format(len(data[:,0])))
    tree = KDTree(x)
    print('| Querying halo Tree')
    ok = np.where((data[:,10]>min_mass)&(data[:,10]<max_mass))
    if(ok[0].size>0):
        rp = np.squeeze(data[ok,4:7])
        res = tree.query_radius(rp,radius)
    else:
        res=[]
    del tree
    return ok,res

def select(config_file):
    __version()
    p = config_selection_obj()
    print('| ------------------------------------------------------------')
    print('| HAST - select_candidate')
    print('| ------------------------------------------------------------')
    try:
        p.parse_input(config_file)
    except:
        print('[Error] {0} file specified cannot be read'.format(config_file))
        sys.exit()
    try:
        sim_zinit = _load_sim(p.output_zinit)
    except IOError:
        print('[Error] {0} file specified cannot be read'.format(p.output_zinit))
        sys.exit()

    try:
        sim_zlast = _load_sim(p.output_zlast)
    except IOError:
        print('[Error] {0} file specified cannot be read'.format(p.output_zlast))
        sys.exit()

    if(p.min_mass>=p.max_mass):
        print('[Error] min_mass>max_mass')
        sys.exit()

    # Sorting the index array
    sim_zinit = sim_zinit[np.argsort(sim_zinit['iord'])]
    sim_zlast = sim_zlast[np.argsort(sim_zlast['iord'])]
    H0 = sim_zlast.properties['h']
    # Computing the Hubble parameter from the Friedmann equation
    z = 1.0/sim_zlast.properties['a']-1.0
    Om = sim_zlast.properties['omegaM0']
    Ol = sim_zlast.properties['omegaL0']
    h = math.sqrt(H0*H0*(Om*math.pow(1+z,3.0)+Ol))
    # Code to physical units
    to_mpc = sim_zlast.properties['boxsize'].in_units('Mpc')*sim_zlast.properties['h']
    to_kpc = 1e3*to_mpc
    # Code to comoving units
    to_mpc_comov = sim_zlast.properties['boxsize'].in_units('Mpc')*sim_zlast.properties['h']/sim_zlast.properties['a']
    to_kpc_comov = 1e3*to_mpc_comov
    scale_m = float(np.sum(sim_zlast.d['mass'].in_units('Msol')))

    print('| ------------------------------------------------------------')
    print('| Selection output = {0} [z={1:5.2f}]'.format(p.output_zlast,abs(1.0/sim_zlast.properties['a']-1.0)))
    print('| Initial output   = {0} [z={1:5.2f}]'.format(p.output_zinit,abs(1.0/sim_zinit.properties['a']-1.0)))
    print('| r_tb             = {0:.2f} R200 '.format(p.rtb))
    print('| r_buffer         = {0:.2f} Mpc'.format(p.rbuffer))
    print('| m_candidate      = {0:.3e} Msol < m < {1:.3e} Msol'.format(p.min_mass,p.max_mass))
    print('| n_neighbors      = {0} < n < {1}'.format(p.min_neighb,p.max_neighb))
    print('| m_neighbor_max   = m < {0:.1e}*m_candidate '.format(p.max_mass_neighb))
    print('| ------------------------------------------------------------')
    sys.stdout.flush()
    rtb = p.rtb
    # Convert rbuffer (Mpc) to kpc for physical-unit positions.
    rbuffer = p.rbuffer * 1e3
    # Get Halo from Ramses clump finder
    if p.halo_finder and p.halo_finder not in ('ramses', 'clump', 'builtin'):
        d = halo_list_pynbody(sim_zlast, p.halo_finder)
    else:
        d = halo_list(p.output_zlast, clump_mass_unit=p.clump_mass_unit)
    candidates,neighbors = find_galaxy(d,rbuffer,p.min_mass,p.max_mass)
    nc = candidates[0].size
    print('| ------------------------------------------------------------')
    print('| Found {0} candidates for {1:.2e}<m<{2:.2e}'.format(nc,p.min_mass,p.max_mass))
    if(nc==0):
        return

    flag = np.zeros(nc)
    xsearch = p.xsearch
    ysearch = p.ysearch
    zsearch = p.zsearch
    rsearch = p.rsearch
    if p.rsearch > 0.0 and np.max(d[:,4:7]) > 1.0:
        try:
            boxsize_kpc = float(sim_zlast.properties['boxsize'].in_units('kpc'))
        except Exception:
            boxsize_kpc = None
        if boxsize_kpc is not None:
            if 0.0 <= xsearch <= 1.0:
                xsearch *= boxsize_kpc
            if 0.0 <= ysearch <= 1.0:
                ysearch *= boxsize_kpc
            if 0.0 <= zsearch <= 1.0:
                zsearch *= boxsize_kpc
            if 0.0 < rsearch <= 1.0:
                rsearch *= boxsize_kpc
    boundary_min = p.boundary_min
    boundary_max = p.boundary_max
    if np.max(d[:,4:7]) > 1.0:
        boxsize = sim_zlast.properties.get('boxsize', None)
        if boxsize is not None:
            try:
                if hasattr(boxsize, "in_units"):
                    boxsize_val = float(boxsize.in_units('kpc'))
                else:
                    boxsize_val = float(boxsize)
                boundary_min = boundary_min * boxsize_val
                boundary_max = boundary_max * boxsize_val
            except Exception:
                pass

    for i in range(nc):
        # Check if neighbors number exceeds cireterion
        if(len(neighbors[i])>p.max_neighb):
            flag[i] = 1
        # Check if neighbors number falls behind cireterion
        if(len(neighbors[i])<p.min_neighb):
            flag[i] = 2
        # Check neigbors mass
        nb = len(neighbors[i])
        for j in range(nb):
            if((d[neighbors[i][j],10]>p.max_mass_neighb*d[candidates[0][i],10])&(neighbors[i][j]!=candidates[0][i])):
                flag[i] = 3
        # Check position
        #if((d[candidates[0][i],4]<rbuffer)or(d[candidates[0][i],4]>1.0-2*rbuffer)):
                #       print "| candidate {0} x = {1} < {2}".format(i,d[candidates[0][i],4],rbuffer)
                #       print "| or            x = {1} > {2}".format(i,d[candidates[0][i],4],1.-2*rbuffer)
            #     flag[i] = 4
        #if((d[candidates[0][i],5]<rbuffer)or(d[candidates[0][i],5]>1.0-2*rbuffer)):
        #    flag[i] = 4
        #if((d[candidates[0][i],6]<rbuffer)or(d[candidates[0][i],6]>1.0-2*rbuffer)):
        #    flag[i] = 4
        if((d[candidates[0][i],4]<boundary_min)or(d[candidates[0][i],4]>boundary_max)):
            print("| candidate {0} x = {1} outside [{2}, {3}]".format(
                i, d[candidates[0][i],4], boundary_min, boundary_max))
            flag[i] = 4
        if((d[candidates[0][i],5]<boundary_min)or(d[candidates[0][i],5]>boundary_max)):
            print("| candidate {0} y = {1} outside [{2}, {3}]".format(
                i, d[candidates[0][i],5], boundary_min, boundary_max))
            flag[i] = 4
        if((d[candidates[0][i],6]<boundary_min)or(d[candidates[0][i],6]>boundary_max)):
            print("| candidate {0} z = {1} outside [{2}, {3}]".format(
                i, d[candidates[0][i],6], boundary_min, boundary_max))
            flag[i] = 4
        if(rsearch>0.0):
            rfilter = math.sqrt((d[candidates[0][i],4]-xsearch)**2+(d[candidates[0][i],5]-ysearch)**2+(d[candidates[0][i],6]-zsearch)**2)
            if(rfilter>rsearch):
                flag[i] = 5

    wh1=np.where(flag==0)
    wh2=np.where(flag==1)
    wh3=np.where(flag==2)
    wh4=np.where(flag==3)
    wh5=np.where(flag==4)
    if(p.rsearch>0.0):
        wh6=np.where(flag==5)
    print('| ------------------------------------------------------------')
    print('| {0:5d} valid candidates'.format(wh1[0].size))
    print('| {0:5d} candidates with n_neighbor>{1}'.format(wh2[0].size,p.max_neighb))
    print('| {0:5d} candidates with n_neighbor<{1}'.format(wh3[0].size,p.min_neighb))
    print('| {0:5d} candidates with m_neighbor>{1:.2f}*m_candidate'.format(wh4[0].size,p.max_mass_neighb))
    print('| {0:5d} candidates close to the box boundaries'.format(wh5[0].size))
    if(p.rsearch>0.0):
        print('| {0:5d} outside of the search region'.format(wh6[0].size))
    print('| ------------------------------------------------------------')
    sys.stdout.flush()
    if(wh1[0].size>0):
        if(p.plot):
            cp = sns.color_palette("Set1",wh1[0].size)
            ax=plot_candidates(d[candidates[0][wh1],:],sim_zlast)
            if((p.plot)and(not p.plot_traceback)):
                pyplot.savefig(p.fname+'.pdf',dpi=100)
            print('| ------------------------------------------------------------')
        print('| Building Tree [{0} particles]'.format(len(sim_zlast)))
        tree = KDTree(np.squeeze((sim_zlast['pos'])),leaf_size=p.tree_nleaves)
        r200 = np.array([])
        print('| Computing Virial radii')
        for i in range(wh1[0].size):
            try:
                rr = pynbody.analysis.halo.virial_radius(sim_zlast,cen=d[candidates[0][wh1[0][i]],4:7],r_max=rbuffer)
            except:
                print('| [Warning] Virial radius computation did not converge')
                rr = 0.
            r200 = np.append(r200,rr)
        print('| Querying particle Tree')
        region_zlast = tree.query_radius(d[candidates[0][wh1],4:7],rtb*r200)
        virial_zlast = tree.query_radius(d[candidates[0][wh1],4:7],r200)
        print('------------------------------------------------------------')
        for i in range(wh1[0].size):
            sys.stdout.flush()
            ind_zlast = sim_zlast['iord'][region_zlast[i]]
            mass_region = float(np.sum(sim_zlast['mass'][region_zlast[i]].in_units('Msol')))
            mass_neighb = np.sum(d[neighbors[wh1[0][i]],10])
            mass_candidate = d[candidates[0][wh1[0][i]],10]
            pos_candidate = np.squeeze(d[candidates[0][wh1[0][i]],4:7])
            # Find those indices at z_init
            region_zinit = np.searchsorted(sim_zinit['iord'],ind_zlast,side='left')
            npart = len(region_zinit)
            print('| {0:3d} | m_candidate={1:.2e} Msol | {2} neighbors | m_region={3:.2e} Msol | npart={4} '.format(i+1,mass_candidate,len(neighbors[wh1[0][i]]),mass_region,npart))
            safety = False
            if((np.max(sim_zinit['x'][region_zinit])-np.min(sim_zinit['x'][region_zinit]))>0.5):
                safety = True
            if((np.max(sim_zinit['y'][region_zinit])-np.min(sim_zinit['y'][region_zinit]))>0.5):
                safety = True
            if((np.max(sim_zinit['z'][region_zinit])-np.min(sim_zinit['z'][region_zinit]))>0.5):
                safety = True
            if(safety):
                print('|     | --- Traceback region lies in boundaries')
                print('| ------------------------------------------------------------')
                continue
            if(r200[i]>0.):
                npart_r200 = len(virial_zlast[i])
            else:
                npart_r200 = 0
            m200 = float(np.sum(sim_zlast['mass'][virial_zlast[i]].in_units('Msol')))
            try:
               # Computing halo spin parameter
               tr = pynbody.analysis.angmom.faceon(sim_zlast[virial_zlast[i]],cen=pos_candidate,cen_size=str(p.rbuffer)+' Mpc')
               lambda200 = pynbody.analysis.angmom.spin_parameter(sim_zlast[virial_zlast[i]])
               tr.revert()
            except:
                lambda200 = 0.0
            xmean = float(np.mean(sim_zinit['x'][region_zinit]))
            ymean = float(np.mean(sim_zinit['y'][region_zinit]))
            zmean = float(np.mean(sim_zinit['z'][region_zinit]))
            print('|     | --- Candidate halo properties')
            print('|     | --------------- m200                   -> {0:.3e} Msol'.format(m200))
            print('|     | --------------- r200                   -> [{0:.1f} kpc phys,{1:.1f} kpc comov, {2:.4f} cu]'.format(r200[i]*to_kpc,r200[i]*to_kpc_comov,r200[i]))
            print('|     | --------------- lambda                 -> {0:.4f}'.format(lambda200))
            print('|     | --------------- npart(r<r200)          -> {0}'.format(npart_r200))
            print('|     | --- Candidate halo position            -> [{0:.5f},{1:.5f},{2:.5f}]'.format(pos_candidate[0],pos_candidate[1],pos_candidate[2]))
            print('|     | --- Mean particle position in ICs      -> [{0:.5f},{1:.5f},{2:.5f}]'.format(xmean,ymean,zmean))
            hull = ConvexHull(sim_zinit['pos'][region_zinit]-sim_zinit['pos'][region_zinit].mean(axis=0))
            if((p.plot)and(p.plot_traceback)):
                proj =[['y','x'],['z','x']]
                dproj =[[5,4],[6,4]]
                for k in range(len(ax)):
                    x=proj[k][0]
                    y=proj[k][1]
                    points_2d = np.squeeze([[sim_zinit[x][region_zinit]],[sim_zinit[y][region_zinit]]]).transpose()
                    hull2d = ConvexHull(points_2d)
                    ax[k].plot(sim_zinit[x][region_zinit][np.append(hull2d.vertices,hull2d.vertices[0])],sim_zinit[y][region_zinit][np.append(hull2d.vertices,hull2d.vertices[0])],'k-',lw=2,color=cp[i])
                    left=np.argmin(sim_zinit[x][region_zinit][hull2d.vertices])
                    #ax[k].annotate(str(i+1),(sim_zinit[x][region_zinit][hull2d.vertices[left]]-0.02,sim_zinit[y][region_zinit][hull2d.vertices[left]]-0.02),fontsize='x-small',color=cp[i])

            print('|     | --- Convex Hull                        -> vol={0:.3e} dens={1:.3e}'.format(hull.volume,float(np.sum(sim_zinit['mass'][region_zinit])/hull.volume)))
            try:
                np.savetxt((p.fname+'_'+str(i+1)).strip(),sim_zinit['pos'][region_zinit][hull.vertices])
                print('|     | --- Particle list outputed to '+(p.fname+'_'+str(i+1)).strip())
            except:
                print('[Error] Cannot write file '+(p.fname+'_'+str(i+1)).strip())
                sys.exit()
            print('| ------------------------------------------------------------')
            sys.stdout.flush()
        if((p.plot)and(p.plot_traceback)):
            pyplot.savefig(p.fname+'.pdf',dpi=100)

    else:
        print('| No haloes matching the criteria')
        return

    return

def decontaminate(config_file):
    __version()
    p = config_decontamination_obj()
    print('| ------------------------------------------------------------')
    print('| HAST - decontaminate it')
    print('| ------------------------------------------------------------')
    try:
        p.parse_input(config_file)
    except:
        print('[Error] {0} file specified cannot be read'.format(config_file))
        sys.exit()


    # Find max output number
    max_out = int(max(glob.glob(p.output_dir+'/output_?????')).split('_')[-1])
    list = sorted(glob.glob(p.output_dir+'/output_?????'))
    nfiles = len(list)

    # Music point shift
    shift = p.ps/2.0**p.pslmin

    # Init
    aexp = np.zeros(max_out)
    x = np.zeros(max_out)
    y = np.zeros(max_out)
    z = np.zeros(max_out)
    m = np.zeros(max_out)
    mnt = np.zeros(max_out)
    mnm = np.zeros(max_out)
    n = np.zeros(max_out)
    idf = np.zeros(max_out)

    region_all_zoom = np.array([]).astype(np.int)
    ncoarse_in_rtb_all = 0

    print('| Search radius     = {0:.2f}*R200'.format(p.rvir_search))
    print('| Traceback radius  = {0:.2f}*R200'.format(p.rvir))
    print('| Halo cut off mass = {0:.2e} Msol'.format(p.halo_cutoff))
    print('| Halo min massfrac = {0:.2e} Msol'.format(p.halo_massfrac))
    print('| Point shift       = {0}'.format(shift))
    print('| ------------------------------------------------------------')
    try:
        print(list[0])
        sim_zinit = _load_sim(list[0])
        sim_zinit = sim_zinit[np.argsort(sim_zinit['iord'])]
    except IOError:
        print('[Error] {0} file specified cannot be read'.format(p.output_zinit))
        sys.exit()
    # Get positions of most massive halo from PHEW halo catalogues
    k = nfiles
    for j in range(nfiles, -1, -1):
        print('| '+p.output_dir+'/output_{j:05d}/clump_{j:05d}.txt?????'.format(j=j))
        print('| ------------------------------------------------------------')
        if not os.path.exists(p.output_dir+'/output_{j:05d}/clump_{j:05d}.txt00001'.format(j=j)):
            print('| clump_{j:05d}.txt????? not found'.format(j=j))
            continue
        else:
            try:
                hl = __halo_list_tracking(list[j-1],p)
                # Find halo with the largest number of cells (i.e. zoomed halo)
                hl = hl[np.flipud(hl[:,3].argsort())]

            except:
                print('| No haloes found in PHEW outputs')
                break
            if(len(hl)==0):
                print('| No haloes found in PHEW outputs')
                continue

            if(j==nfiles):
                if((p.halo_coords[0]>0.) & (p.halo_coords[1]>0.) & (p.halo_coords[2]>0.)):
                    dist_halo = np.sqrt(np.power(hl[:,4]-p.halo_coords[0],2)+np.power(hl[:,5]-p.halo_coords[1],2)+np.power(hl[:,6]-p.halo_coords[2],2))
                    # Selected halo is the closest coordinate
                    id = np.argmin(dist_halo)
                else:
                    id = p.halo_num-1
                    print('| Selecting halo ranked ',p.halo_num,' with ',int(hl[id,3]),' cells')
            # Build tree for halos
            tree_halo = KDTree(np.squeeze((hl[:,4:7])),leaf_size=p.tree_nleaves)
            diff = k-j+1
            if(j<nfiles):
                # Save previous snapshot
                sim_prev = sim_curr
                tree_part_prev = tree_part_curr
                # Find halos matching coordinate filter around previous halo
                halo_candidates = tree_halo.query_radius([x[k],y[k],z[k]],p.rvir_search*r200_start)[0]
                # Load current snapshot
                sim_curr = _load_sim(list[j-1])
                sim_curr = sim_curr[np.argsort(sim_curr['iord'])]
                aexp_curr = float(sim_curr.properties['a'])
                to_msol = float(np.sum(sim_curr.d['mass'].in_units('Msol')))
                mass_cutoff = max(p.halo_cutoff/to_msol,p.halo_massfrac*mass_curr)
                # Filter low mass halos
                halo_candidates = halo_candidates[hl[halo_candidates,10]>mass_cutoff]
                if(len(halo_candidates)==0):
                    print('| No halos found')
                    print('| Tracking stopped at aexp={0}'.format(aexp_curr))
                    break
                # Gather particles in the previous selected halo
                halo_part_prev = tree_part_prev.query_radius([x[k],y[k],z[k]],p.rvir_track*r200_start)[0]
                # Build tree for particles
                print('|    | npart tree               = {0:9d} ------------------'.format(len(sim_curr)))
                tree_part_curr = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
                #print '|    | Previous halo population fractions --------------------'
                print('|    | Previous halo population = {0:7d} --------------------'.format(len(halo_candidates)))
                print('|    | Cutoff mass              = {0:4.2e} -------------------'.format(mass_cutoff*to_msol))

                ids_frac = np.array([])
                # Looping over candidates
                ids_frac = np.zeros(len(halo_candidates))
                ii = 0
                for halo in halo_candidates:

                    # Compute R200 in code units
                    # M200 = 4/3*pi*200*rho_mean*R200^3
                    # In code units, rho_mean=1
                    r200_candidate = (hl[halo,10]*3./(200.*4.*math.pi))**(1.0/3.0)
                    # Gather particle of the halo to track
                    halo_part_curr = tree_part_curr.query_radius(hl[halo,4:7],p.rvir_track*r200_candidate)[0]
                    # Match unique indices
                    matching_ids = np.where(np.in1d(sim_curr['iord'][halo_part_curr],sim_prev['iord'][halo_part_prev]))[0]
                    # Matching indices fraction
                    ids_frac[ii] = float(len(matching_ids))/float(len(halo_part_prev))
                    ii += 1
                    print('|    |         halo {0:7d} | idf={1:5.2f}% | m={2:5.2e} Msol'.format(halo,100*ids_frac[ii-1],hl[halo,10]*to_msol))
                # Selecting best candidate
                best_candidate = np.argmax(ids_frac)
                id = halo_candidates[best_candidate]
                halo_rejected = np.delete(halo_candidates,best_candidate)
                # Computing Virial radius of the best candidate
                try:
                    # Computing R200 from particle distribution
                    r200_curr = pynbody.analysis.halo.virial_radius(sim_curr,cen=hl[id,4:7],r_max=p.rvir_search*r200_curr)
                except:
                    # Computing R200 from clumpfinder's mass
                    print('| [Warning] Virial radius computation did not converge')
                    r200_curr = (hl[id,10]*3./(200.*4.*math.pi))**(1.0/3.0)
                mass_curr = hl[id,10]
                print('|    |    -->  halo {0:7d} selected'.format(id))
                print('| ------------------------------------------------------------')

            # Final snapshot - starting point
            else:
                print('| Closest halo coordinates  = [{0:.5f},{1:.5f},{2:.5f}] cu'.format(hl[id,4],hl[id,5],hl[id,6]))
                if((p.halo_coords[0]>0.) & (p.halo_coords[1]>0.) & (p.halo_coords[2]>0.)):
                    print('| Relative distance         = {0:.2e} cu'.format(np.min(dist_halo)))
                # Loading first snapshot
                sim_curr = _load_sim(list[j-1])
                aexp_curr = float(sim_curr.properties['a'])
                # Computing virial radius
                r200_start = pynbody.analysis.halo.virial_radius(sim_curr.d,cen=hl[id,4:7],r_max=0.5)
                r200_curr = r200_start
                mass_curr = hl[id,10]
                id_start = id
                # Code to physical units
                to_mpc = sim_curr.properties['boxsize'].in_units('Mpc')*sim_curr.properties['h']
                to_kpc = 1e3*to_mpc
                to_msol = float(np.sum(sim_curr.d['mass'].in_units('Msol')))
                print('| R200                      = {0:.4f} kpc'.format(r200_start*to_kpc))
                print('| M200                      = {0:.2e} Msol'.format(hl[id,10]*to_msol))
                print('| coords                    = {0} cu'.format(hl[id,4:7]))
                sim_curr = sim_curr[np.argsort(sim_curr['iord'])]
                tree_part_curr = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
                print('| ------------------------------------------------------------')
                ids_frac = 1.0
                # Find halos matching coordinate filter around previous halo
                halo_candidates = tree_halo.query_radius(np.squeeze(hl[id,4:7]),p.rvir_search*r200_curr)[0]
                # Filter low mass halos
                halo_candidates = halo_candidates[hl[halo_candidates,10]*to_msol>p.halo_cutoff]
                # Selected halo
                best_candidate = np.where(halo_candidates==id)[0]
                halo_rejected = np.delete(halo_candidates,best_candidate)


            # Code to physical units
            to_mpc = sim_curr.properties['boxsize'].in_units('Mpc')*sim_curr.properties['h']
            to_kpc = 1e3*to_mpc
            # Code to comoving units
            to_mpc_comov = sim_curr.properties['boxsize'].in_units('Mpc')*sim_curr.properties['h']/sim_curr.properties['a']
            to_kpc_comov = 1e3*to_mpc_comov
            # Find zoomed particles
            zoom_part = np.where(sim_curr['mass']<1.1*np.min(sim_curr['mass']))
            # Find coarse particles
            coarse_part = np.where(sim_curr['mass']>1.1*np.min(sim_curr['mass']))
            # Look for particles contaminating the zoom region
            tree = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
            virial_curr = tree_part_curr.query_radius(hl[id,4:7].reshape(1,-1),r200_curr)[0]
            region_curr = tree_part_curr.query_radius(hl[id,4:7].reshape(1,-1),p.rvir*r200_curr)[0]
            # Include all the zoom particles
            region_curr_zoom = np.unique(np.append(zoom_part,region_curr))
            region_all_zoom = np.unique(np.append(region_all_zoom,region_curr_zoom))
            m200 = float(np.sum(sim_curr['mass'][virial_curr].in_units('Msol')))
            mass_candidate = hl[id,10]
            coarse_in_rtb = np.where(sim_curr['mass'][region_curr]>1.1*np.min(sim_curr['mass']))
            ncoarse_in_rtb_all += len(coarse_in_rtb[0])
            coarse_in_r200 = np.where(sim_curr['mass'][virial_curr]>1.1*np.min(sim_curr['mass']))
            # Compute halo spin parameter
            print('| R200                      = {0:.1f} kpc physical / {1:.1f} kpc comoving'.format(r200_curr*to_kpc,r200_curr*to_kpc_comov))
            print('| M200                      = {0:.2e} Msol'.format(m200))
            #print '| Spin parameter            = {0:.3f}'.format(lambda200)
            print('| M_clump                   = {0:.2e} Msol'.format(mass_candidate))
            print('| position                  = [{0:.4f},{1:.4f},{2:.4f}]'.format(hl[id,4],hl[id,5],hl[id,6]))
            print('| ------------------------------------------------------------')
            print('| npart_tot(r<R200)         = {1}'.format(p.rvir,len(virial_curr)))
            print('| npart_tot(r<{0}*R200)     = {1}'.format(p.rvir,len(region_curr)))
            print('| npart_coarse(r<{0}*R200)  = {1}'.format(p.rvir,len(coarse_in_rtb[0])))
            print('| npart_coarse_all          = {0}'.format(ncoarse_in_rtb_all))
            print('| contamination(r<{0}*R200) = {1:.1f}%'.format(p.rvir,100*float(np.sum(sim_curr['mass'][region_curr][coarse_in_rtb]))/float(np.sum(sim_curr['mass'][region_curr]))))
            print('| npart_zoom                = {0}'.format(len(zoom_part[0])))
            print('| npart_tot                 = {0}'.format(len(sim_curr)))
            # Get unique indices
            ind_curr = sim_curr['iord'][region_all_zoom]
            # Trace indices back in the initial output
            region_zinit = np.searchsorted(sim_zinit['iord'],ind_curr,side='left')
            # Find coarse particles in the zoom region at z_init
            coarse_in_rtb_init = np.where(sim_zinit['mass'][region_zinit]>1.1*np.min(sim_zinit['mass']))
            # Find zoom particles at z_init
            zoom_in_rtb_init = np.where(sim_zinit['mass'][region_zinit]<1.1*np.min(sim_zinit['mass']))
            zoom_init = np.where(sim_zinit['mass']<1.1*np.min(sim_zinit['mass']))
            # Computing center of the zoom particles in z_init
            zinit_center = [
                np.average(sim_zinit['x'][region_zinit][zoom_in_rtb_init]),
                np.average(sim_zinit['y'][region_zinit][zoom_in_rtb_init]),
                np.average(sim_zinit['z'][region_zinit][zoom_in_rtb_init])]
            # Centering
            sim_zinit['pos'] = sim_zinit['pos']-zinit_center
            allowed = np.where((sim_zinit['r'][region_zinit]<p.rexclude)|(sim_zinit['mass'][region_zinit]<1.1*np.min(sim_zinit['mass'])))
            not_allowed = np.where((sim_zinit['r'][region_zinit]>=p.rexclude)&(sim_zinit['mass'][region_zinit]>1.1*np.min(sim_zinit['mass'])))
            print('| Included coarse part      = {0}'.format(len(allowed[0])))
            print('| Excluded coarse part      = {0}'.format(len(not_allowed[0])))
            sim_zinit['pos'] = sim_zinit['pos']+zinit_center
            if(len(coarse_in_rtb_init)>0):
                try:
                    sim_curr['pos'] = sim_curr['pos']-hl[id,4:7]
                    print('| r_min coarse part/R200    = {0:.3e}'.format(float(np.min(sim_curr['r'][region_curr][coarse_in_rtb]))/r200_curr))
                    print('| r_mean coarse part/R200   = {0:.3e}'.format(float(np.mean(sim_curr['r'][region_curr][coarse_in_rtb]))/r200_curr))
                    sim_curr['pos'] = sim_curr['pos']+hl[id,4:7]
                except:
                    pass
                # Computing convex hulls volumes
                hull = ConvexHull(sim_zinit['pos'][region_zinit][allowed])
                hull_zoom = ConvexHull(sim_zinit['pos'][zoom_init])
                print('| Convex Hull coarse part -> vol={0:.3e} dens={1:.3e}'.format(hull.volume,float(np.sum(sim_zinit['mass'][region_zinit][allowed])/hull.volume)))
                print('| Convex Hull zoom part   -> vol={0:.3e} dens={1:.3e}'.format(hull_zoom.volume,float(np.sum(sim_zinit['mass'][zoom_init])/hull_zoom.volume)))
                print('| Volume increase         -> {0:.2f}%'.format(100*(hull.volume/hull_zoom.volume)-100.))


            if((np.max(ids_frac)>0.01)&(aexp_curr>p.aexp_min)):
                x[j-1] = hl[id,4]
                y[j-1] = hl[id,5]
                z[j-1] = hl[id,6]
                m[j-1] = hl[id,10]*to_msol
                n[j-1] = hl[id,3]
                if(len(halo_rejected)>0):
                    mnt[j-1] = np.sum(hl[halo_rejected,10])*to_msol
                    mnm[j-1] = np.max(hl[halo_rejected,10])*to_msol
                else:
                    mnt[j-1] = 0.0
                    mnm[j-1] = 0.0
                    idf[j-1] = np.max(ids_frac)
                aexp[j-1] = aexp_curr
                k = j-1
            else:
                print('| ------------------------------------------------------------')
                print('| Tracking stopped at aexp={0}'.format(aexp_curr))
                break


    print('| ------------------------------------------------------------')
    if(len(coarse_in_rtb_init)>0):
        try:
            np.savetxt((p.fname).strip()+'_part',sim_zinit['pos'][region_zinit][allowed][hull.vertices]-shift)
            print('| Particle list outputed to '+(p.fname).strip())
        except:
            print('[Error] Cannot write file '+(p.fname).strip())
            sys.exit()

        sys.stdout.flush()
    else:
        print('| No contamination')

    # Remove NaNs
    defined = np.where(aexp>0.0)
    x = x[defined]
    y = y[defined]
    z = z[defined]
    m = m[defined]
    n = n[defined]
    mnm = mnm[defined]
    mnt = mnt[defined]
    aexp = aexp[defined]
    idf = idf[defined]

    # Write results
    np.savetxt(p.fname+'_track',np.transpose(np.squeeze([aexp,x,y,z,m,n,idf,mnt,mnm])),header="aexp x y z mass npart ids_fraction mass_neighb_max mass_neighb_tot")

    # Fit coefficients
    cx = polyfit(aexp, x, 3, full=True, w=m)[0]
    cy = polyfit(aexp, y, 3, full=True, w=m)[0]
    cz = polyfit(aexp, z, 3, full=True, w=m)[0]

    # Print result
    print('| ------------------------------------------------------------')
    print('| RAMSES polynomial coefficients for camera halo tracking')
    print('| ------------------------------------------------------------')
    print('| xcentre_frame='+','.join('{:6f}'.format(i) for i in cx))
    print('| ycentre_frame='+','.join('{:6f}'.format(i) for i in cy))
    print('| zcentre_frame='+','.join('{:6f}'.format(i) for i in cz))

    # Plotting
    if p.plot:
        print('| ------------------------------------------------------------')
        print('| Plotting')
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        cp = sns.color_palette(flatui)
        sns.set_context('poster')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})

        # Plotting tracked coordinates and fitted polynome
        fig,ax = pyplot.subplots(1)
        ax.plot(aexp, x, 'o', c=cp[0], ms=5)
        ax.plot(aexp, cx[0]+cx[1]*aexp+cx[2]*aexp**2+cx[3]*aexp**3, c=cp[0], lw=3, label='x')
        ax.plot(aexp, y, 'o', c=cp[1], ms=5)
        ax.plot(aexp, cy[0]+cy[1]*aexp+cy[2]*aexp**2+cy[3]*aexp**3, c=cp[1], lw=3, label='y')
        ax.plot(aexp, z, 'o', c=cp[2], ms=5)
        ax.plot(aexp, cz[0]+cz[1]*aexp+cz[2]*aexp**2+cz[3]*aexp**3, c=cp[2], lw=3, label='z')
        ax.set_xlabel('aexp')
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.legend()
        pyplot.savefig(p.fname+".pdf")
        pyplot.close(fig)

        # Plotting mass evolution
        fig,ax = pyplot.subplots(1)
        ax.plot(aexp, np.log10(m), '-', c=cp[0],label='tracked halo')
        ax.plot(aexp, np.log10(mnm), '-', c=cp[1], label='heaviest companion')
        ax.plot(aexp, np.log10(mnt), '-', c=cp[2], label='total companion')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlim([0.0,1.0])
        ax.legend()
        ax.set_xlabel('aexp')
        ax.set_ylabel(r'Mass [M$_{\odot}$]')
        pyplot.savefig(p.fname+'_mass.pdf')
        pyplot.close(fig)

        # Reload last output
        sim_zlast = _load_sim(list[-1])
        sim_zlast = sim_zlast[np.argsort(sim_zlast['iord'])]
        hl = __halo_list_tracking(list[-1],p)
        # Find zoomed particles
        zoom_part = np.where(sim_zlast['mass']<1.1*np.min(sim_zlast['mass']))
        cp = sns.color_palette(flatui)
        center = [0.,0.,0.]
        sns.set_style("ticks",{"axes.grid": False,"xtick.direction":'in',"ytick.direction":'in'})
        fig,ax = pyplot.subplots(1,2,figsize=(16,8))
        proj =[['x','y'],['x','z']]
        dproj =[[4,5],[4,6]]
        for i in range(len(ax)):
            x=proj[i][0]
            y=proj[i][1]
            try:
                xmin_coarse_in_rtb = float(np.min(sim_zinit.d[x][region_zinit][coarse_in_rtb_init]))
                ymin_coarse_in_rtb = float(np.min(sim_zinit.d[y][region_zinit][coarse_in_rtb_init]))
                xmax_coarse_in_rtb = float(np.max(sim_zinit.d[x][region_zinit][coarse_in_rtb_init]))
                ymax_coarse_in_rtb = float(np.max(sim_zinit.d[y][region_zinit][coarse_in_rtb_init]))
            except:
                xmin_coarse_in_rtb = 1.0
                ymin_coarse_in_rtb = 1.0
                xmax_coarse_in_rtb = 0.0
                ymax_coarse_in_rtb = 0.0

            xmin = min(xmin_coarse_in_rtb,float(np.min(sim_zlast.d[x][zoom_part])))-0.01
            ymin = min(ymin_coarse_in_rtb,float(np.min(sim_zlast.d[y][zoom_part])))-0.01
            xmax = max(xmax_coarse_in_rtb,float(np.max(sim_zlast.d[x][zoom_part])))+0.01
            ymax = max(ymax_coarse_in_rtb,float(np.max(sim_zlast.d[y][zoom_part])))+0.01
            pmin = min(xmin,ymin)
            pmax = max(xmax,ymax)
            ax[i].set_xlim([pmin,pmax])
            ax[i].set_ylim([pmin,pmax])
            ax[i].set_xlabel(x)
            ax[i].set_ylabel(y)
            im,xedges,yedges = np.histogram2d(sim_zlast.d[x][zoom_part],sim_zlast.d[y][zoom_part],
                weights=sim_zlast.d['mass'][zoom_part],bins=1024,range=[[pmin,pmax],[pmin,pmax]])
            im = np.rot90(im)
            # Plotting 2D Convex Hull
            points_2d = np.squeeze([[sim_zinit[x][region_zinit][allowed]],
                [sim_zinit[y][region_zinit][allowed]]]).transpose()
            hull2d = ConvexHull(points_2d)
            ax[i].plot(sim_zinit[x][region_zinit][allowed][np.append(hull2d.vertices,hull2d.vertices[0])],
                sim_zinit[y][region_zinit][allowed][np.append(hull2d.vertices,hull2d.vertices[0])],
                'k-',lw=1.,color=cp[5],label='Lagrangian volume')
            #points_2d = np.squeeze([[sim_zinit[x][region_zinit][zoom_in_rtb_init]],[sim_zinit[y][region_zinit][zoom_in_rtb_init]]]).transpose()
            points_2d = np.squeeze([[sim_zinit[x][zoom_init]],[sim_zinit[y][zoom_init]]]).transpose()
            hull2d = ConvexHull(points_2d)
            ax[i].plot(sim_zinit[x][zoom_init][np.append(hull2d.vertices,hull2d.vertices[0])],
                sim_zinit[y][zoom_init][np.append(hull2d.vertices,hull2d.vertices[0])],
                'k-',lw=1.,color=cp[3],label='Lagrangian volume zoom part')
            # Plot main zoomed halo center
            h1 = ax[i].scatter(hl[id_start,dproj[i][0]],hl[id_start,dproj[i][1]],c=cp[0],alpha=0.35)
            h2 = ax[i].scatter(zinit_center[dproj[i][0]-4],zinit_center[dproj[i][1]-4],c=cp[1],alpha=0.35,zorder=10)
            # Plot R200 circle
            an = np.linspace(0,2*np.pi,100)
            ax[i].plot(r200_start*np.cos(an)+hl[id_start,dproj[i][0]],r200_start*np.sin(an)+hl[id_start,dproj[i][1]],color=cp[1],label='R200',lw=1.)
            ax[i].plot(p.rvir*r200_start*np.cos(an)+hl[id_start,dproj[i][0]],p.rvir*r200_start*np.sin(an)+hl[id_start,dproj[i][1]],color=cp[2],label='Rtb',lw=1.)
            ax[i].plot(p.rexclude*np.cos(an)+zinit_center[dproj[i][0]-4],p.rexclude*np.sin(an)+zinit_center[dproj[i][1]-4],color='k',label='Rexclude',lw=1.)
            # Plot contaminating particles
            if(ncoarse_in_rtb_all>0):
                # Plot contaminating particles at zinit
                # Plot only unique particle [x,y] downsampled positions to lower plot size in memory
                points_2d = np.vstack((sim_zinit[x][region_zinit][coarse_in_rtb_init],sim_zinit[y][region_zinit][coarse_in_rtb_init])).transpose()
                points_2d = np.round(points_2d*2000.)/2000.
                unique_points_2d = __unique_rows(points_2d)
                ax[i].scatter(unique_points_2d[:,0],unique_points_2d[:,1],
                    c=cp[4],marker='+',s=5,alpha=0.50,linewidth=0.5,label='Contaminating part initial')
            #ax[i].set(adjustable='box-forced', aspect='equal')
            tv = ax[i].imshow(np.log10(im),cmap='bone_r',interpolation='quadric',aspect='equal',extent=[pmin,pmax,pmin,pmax])
        ax[0].legend(loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.10, 1.0, 0.1),ncol=2,markerscale=5.)
        out=p.fname+'_decontamination.pdf'
        pyplot.savefig(out,dpi=100)

def analyse(config_file):
    __version()
    p = config_analysis_obj()
    print('| ------------------------------------------------------------')
    print('| HAST - halo analysis')
    print('| ------------------------------------------------------------')
    p.parse_input(config_file)
    try:
        p.parse_input(config_file)
    except:
        print('[Error] {0} file specified cannot be read'.format(config_file))
        sys.exit()

    try:
        sim = _load_sim(p.output)
    except IOError:
        print('[Error] {0} file specified cannot be read'.format(p.output))
        sys.exit()

    index_zlast = int(p.output.split('_')[-1])

    print('| Output = {0} [z={1:5.2f}]'.format(p.output,abs(1.0/sim.properties['a']-1.0)))

    # Code to physical units
    to_mpc = sim.properties['boxsize'].in_units('Mpc')*sim.properties['h']
    to_kpc = 1e3*to_mpc
    h0 = sim.properties["h"]*1e2
    # Code to comoving units
    to_mpc_comov = sim.properties['boxsize'].in_units('Mpc')*sim.properties['h']/sim.properties['a']
    to_kpc_comov = 1e3*to_mpc_comov
    rbuffer = p.rbuffer*sim.properties['a']/(sim.properties['h']*sim.properties['boxsize'].in_units('Mpc'))
    # Find zoomed particles
    zoom_part = np.where(sim.d['mass']<1.1*np.min(sim.d['mass']))
    # Find coarse particles
    coarse_part = np.where(sim.d['mass']>1.1*np.min(sim.d['mass']))
    # Read clump finder results
    d = __halo_list_tracking(p.output,p)
    halo_zoom = int(p.halo_num-1)
    # Compute R200 for the zoomed halo
    print('| Computing virial radius')
    try:
        sim.g['mass']
        r200 = pynbody.analysis.halo.virial_radius(sim,cen=d[halo_zoom,4:7],r_max=rbuffer)
    except:
        print('| [Error] Virial radius computation did not converge')
        # Centering on target halo
        sim['pos'] = sim['pos']-d[halo_zoom,4:7]
        # Computing radial mass profile
        mhist,ed = np.histogram(sim['r'],range=(0.0,rbuffer),bins=512,weights=sim['mass'])
        sim['pos'] = sim['pos']+d[halo_zoom,4:7]
        # Shell volume
        vol_bin = (4.0/3.0)*math.pi*(ed[1:]**3-ed[:-1]**3)
        # Bin radius
        r_bin = ed[0:-1]+0.5*(ed[2]-ed[1])
        # Computing density
        rho = mhist/vol_bin
        # Critical density where r=r200
        overdens = 178.0
        rho_target = overdens*sim.properties["omegaM0"]*pynbody.analysis.cosmology.rho_crit(sim,z=0)*(1.0+sim.properties["z"])**3
        ivirial = np.argmin(np.abs(rho-rho_target))
        r200 = r_bin[ivirial]

    print('| R200     = {0:.1f} kpc physical / {1:.1f} kpc comoving'.format(r200*to_kpc,r200*to_kpc_comov))
    tr = pynbody.analysis.angmom.faceon(sim,cen=d[halo_zoom,4:7],cen_size=str(0.5*r200*to_kpc)+' kpc')
    lambda200 = pynbody.analysis.angmom.spin_parameter(sim.d[sim.d['r']<r200])
    tr.revert()
    print('| Lambda = {0:.3f}'.format(lambda200))
    # Look for particles contaminating the zoom region
    print('| Building Tree - dark matter [{0:10d} particles]'.format(len(sim.d)))
    tree_d = KDTree(np.squeeze((sim.d['pos'])),leaf_size=p.tree_nleaves)
    region_d = tree_d.query_radius(d[halo_zoom,4:7].reshape(1,-1),r200*p.rvr)[0]
    sim_d = sim.d[region_d]
    del tree_d,region_d
    gc.collect()
    if(len(sim.g)>0):
        print('| Building Tree - gas         [{0:10d} particles]'.format(len(sim.g)))
        tree_g = KDTree(np.squeeze((sim.g['pos'])),leaf_size=p.tree_nleaves)
        region_g = tree_g.query_radius(d[halo_zoom,4:7].reshape(1,-1),r200*p.rvr)[0]
        sim_g = sim.g[region_g]
        del tree_g,region_g
        gc.collect()
        print('| Building Tree - stars       [{0:10d} particles]'.format(len(sim.s)))
        tree_s = KDTree(np.squeeze((sim.s['pos'])),leaf_size=p.tree_nleaves)
        region_s = tree_s.query_radius(d[halo_zoom,4:7].reshape(1,-1),r200*p.rvr)[0]
        sim_s = sim.s[region_s]
        del tree_s,region_s
        gc.collect()

    m_dm = float(np.sum(sim_d['mass'].in_units('Msol')))
    n_dm = len(sim_d)
    coarse_in_rtb = np.where(sim_d['mass']>1.1*np.min(sim_d['mass']))
    print('| M_dm     = {0:.3e}'.format(m_dm))
    print('| n_dm     = {0}'.format(n_dm))
    if(len(sim.g)>0):
        m_gas = float(np.sum(sim_g['mass'].in_units('Msol')))
        n_gas = len(sim_g)
        m_stars = float(np.sum(sim_s['mass'].in_units('Msol')))
        n_stars = len(sim_s)
        print('| M_gas    = {0:.3e}'.format(m_gas))
        print('| n_gas    = {0}'.format(n_gas))
        print('| M_stars  = {0:.3e}'.format(m_stars))
        print('| n_stars  = {0}'.format(n_stars))

        tform = (sim.s['tform'])/(h0*1e5/3.08e24)/(365.*24.*3600.*1e9)
        sfr,t = np.histogram(tform,bins=p.nbin_sfr,weights=sim.s["mass"].in_units("Msol")/1e9)
        sfr /= (t[2]-t[1])
        t = t[0:-1]+0.5*(t[2]-t[1])
        ok = np.where(sfr>0.0)
        sfr = sfr[ok]
        t = t[ok]
        print('| mean SFR   = {0:.2e}'.format(np.mean(sfr)))
        print('| median SFR = {0:.2e}'.format(np.median(sfr)))
        print('| min SFR    = {0:.2e}'.format(np.min(sfr)))
        print('| max SFR    = {0:.2e}'.format(np.max(sfr)))
        if(p.plot):
            sns.set_context("poster")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            fig,ax1=pyplot.subplots(1, figsize=(10,6))
            fig.subplots_adjust(wspace=0)
            fig.subplots_adjust(hspace=0.075)
            pyplot.gcf().subplots_adjust(bottom=0.15)
            ax1.plot(t,sfr,color=cp[0])
            ax1.locator_params(nbins=3, axis='x')
            ax1.set_xlabel(r'Lookback time [Gyr]')
            tticks = np.linspace(-13.75,0.99*np.max(t),6)
            zticks = np.array([])
            for i in range(len(tticks)):
                zticks = np.append(zticks,z_at_value(Planck15.age, (tticks[i]+Planck15.age(0.0).value)*u.Gyr))
            ax2 = ax1.twiny()
            ax1.set_xticks(tticks)
            ax2.set_xticks(tticks)
            ax2.set_xlabel('z')
            ax1.set_xlim([-13.75,np.max(tticks)])
            ax2.set_xlim([-13.75,np.max(tticks)])
            zticklabels =  ["%.1f" % x for x in zticks]
            tticklabels =  ["%.1f" % x for x in tticks]
            ax1.set_xticklabels(tticklabels)
            ax2.set_xticklabels(zticklabels)
            ax2.grid(False)
            ax1.set_ylabel(r'SFR [$M_{\odot}/yr$]')
            pyplot.savefig('analysis_sfr.pdf')

    if(p.plot):
        print('| Center     = ',(d[halo_zoom,4:7]))
        print('| Plotting DM')
        center = [0.,0.,0.]
        sns.set_context('poster')
        sns.set_style("ticks",{"axes.grid": False,"xtick.direction":'in',"ytick.direction":'in'})
        fig,ax = pyplot.subplots(1,2,figsize=(16,8))
        proj =[['x','y'],['x','z']]
        dproj =[[4,5],[4,6]]
        for i in range(len(ax)):
            x=proj[i][0]
            y=proj[i][1]
            xmin = float(np.min(sim.d[x][zoom_part]))-0.01
            ymin = float(np.min(sim.d[y][zoom_part]))-0.01
            xmax = float(np.max(sim.d[x][zoom_part]))+0.01
            ymax = float(np.max(sim.d[y][zoom_part]))+0.01
            pmin = min(xmin,ymin)
            pmax = max(xmax,ymax)
            ax[i].set_xlim([pmin,pmax])
            ax[i].set_ylim([pmin,pmax])
            ax[i].set_xlabel(x)
            ax[i].set_ylabel(y)
            im,xedges,yedges = np.histogram2d(sim.d[x][zoom_part],sim.d[y][zoom_part],weights=sim.d['mass'][zoom_part],bins=1024,range=[[pmin,pmax],[pmin,pmax]])
            im = np.rot90(im)
            h1 = ax[i].scatter(d[halo_zoom,dproj[i][0]],d[halo_zoom,dproj[i][1]],c=cp[0],alpha=0.10)
            h1 = ax[i].scatter(d[halo_zoom,dproj[i][1:10]],d[halo_zoom,dproj[i][1]],c=cp[1],alpha=0.10)
            # Plot R200 circle
            an = np.linspace(0,2*np.pi,100)
            ax[i].plot(r200*np.cos(an)+d[halo_zoom,dproj[i][0]],r200*np.sin(an)+d[halo_zoom,dproj[i][1]],color=cp[3],label='R200',lw=1.)
            tv = ax[i].imshow(np.log10(im),cmap='bone_r',interpolation='quadric',aspect='equal',extent=[pmin,pmax,pmin,pmax])
        ax[0].legend(loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.10, 1.0, 0.1),ncol=2,markerscale=5.)
        out='analysis_dm.pdf'
        pyplot.savefig(out,dpi=100)
        pyplot.close(fig)

        print('| Plotting stars')
        if(len(sim.s)>0):
            center = [0.,0.,0.]
            fig,ax = pyplot.subplots(1,2,figsize=(16,8))
            proj =[['x','y'],['x','z']]
            dproj =[[4,5],[4,6]]
            for i in range(len(ax)):
                x=proj[i][0]
                y=proj[i][1]
                xmin = float(np.min(sim.s[x][zoom_part]))-0.01
                ymin = float(np.min(sim.s[y][zoom_part]))-0.01
                xmax = float(np.max(sim.s[x][zoom_part]))+0.01
                ymax = float(np.max(sim.s[y][zoom_part]))+0.01
                pmin = min(xmin,ymin)
                pmax = max(xmax,ymax)
                ax[i].set_xlim([pmin,pmax])
                ax[i].set_ylim([pmin,pmax])
                ax[i].set_xlabel(x)
                ax[i].set_ylabel(y)
                im,xedges,yedges = np.histogram2d(sim.s[x],sim.s[y],weights=sim.s['mass'],bins=1024,range=[[pmin,pmax],[pmin,pmax]])
                im = np.rot90(im)
                h1 = ax[i].scatter(d[halo_zoom,dproj[i][0]],d[halo_zoom,dproj[i][1]],c=cp[0],alpha=0.10)
                # Plot R200 circle
                an = np.linspace(0,2*np.pi,100)
                ax[i].plot(r200*np.cos(an)+d[halo_zoom,dproj[i][0]],r200*np.sin(an)+d[halo_zoom,dproj[i][1]],color=cp[2],label='R200',lw=1.)
                tv = ax[i].imshow(np.log10(im),cmap='bone_r',interpolation='quadric',aspect='equal',extent=[pmin,pmax,pmin,pmax])
            ax[0].legend(loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.10, 1.0, 0.1),ncol=2,markerscale=5.)
            out='analysis_stars.pdf'
            pyplot.savefig(out,dpi=100)
            pyplot.close(fig)


def track(config_file):
    __version()
    p = config_tracking_obj()
    print('| ------------------------------------------------------------')
    print('| HAST - halo tracking')
    print('| ------------------------------------------------------------')
    p.parse_input(config_file)

    print('| Fitting coefficients for: '+p.output_dir)


    # Find max output number
    max_out = int(max(glob.glob(p.output_dir+'/output_?????')).split('_')[-1])
    list = sorted(glob.glob(p.output_dir+'/output_?????'))
    nfiles = len(list)

    # Init
    aexp = np.zeros(max_out)
    x = np.zeros(max_out)
    y = np.zeros(max_out)
    z = np.zeros(max_out)
    m = np.zeros(max_out)
    mnt = np.zeros(max_out)
    mnm = np.zeros(max_out)
    n = np.zeros(max_out)
    idf = np.zeros(max_out)

    print('| Search radius     = {0:.2f}*R200'.format(p.rvir_search))
    print('| Traceback radius  = {0:.2f}*R200'.format(p.rvir))
    print('| Halo coordinates  = {0}'.format(p.halo_coords))
    print('| Halo cut off mass = {0:.2e} Msol'.format(p.halo_cutoff))
    print('| ------------------------------------------------------------')

    # Get positions of most massive halo from PHEW halo catalogues
    k = nfiles
    for j in range(nfiles, -1, -1):
        print('| '+p.output_dir+'/output_{j:05d}/clump_{j:05d}.txt?????'.format(j=j))
        print('| ------------------------------------------------------------')
        if not os.path.exists(p.output_dir+'/output_{j:05d}/clump_{j:05d}.txt00001'.format(j=j)):
            print('| clump_{j:05d}.txt????? not found'.format(j=j))
            continue
        else:
            try:
                hl = __halo_list_tracking(list[j-1],p)
            except:
                print('| No haloes found in PHEW outputs')
                break
            if(len(hl)==0):
                print('| No haloes found in PHEW outputs')
                continue
            id = -1
            dist_halo = np.sqrt(np.power(hl[:,4]-p.halo_coords[0],2)+np.power(hl[:,5]-p.halo_coords[1],2)+np.power(hl[:,6]-p.halo_coords[2],2))
            # Build tree for halos
            tree_halo = KDTree(np.squeeze((hl[:,4:7])),leaf_size=p.tree_nleaves)
            # Selected halo is the closest coordinate
            id = np.argmin(dist_halo)
            diff = k-j+1
            if(j<nfiles):
                # Save previous snapshot
                sim_prev = sim_curr
                tree_part_prev = tree_part_curr
                # Find halos matching coordinate filter around previous halo
                halo_candidates = tree_halo.query_radius([x[k],y[k],z[k]],p.rvir_search*r200_start)[0]

                # Load current snapshot
                sim_curr = _load_sim(list[j-1])
                aexp_curr = float(sim_curr.properties['a'])
                to_msol = float(np.sum(sim_curr.d['mass'].in_units('Msol')))
                mass_cutoff = max(p.halo_cutoff/to_msol,p.halo_massfrac*mass_curr)
                # Filter low mass halos
                halo_candidates = halo_candidates[hl[halo_candidates,10]>mass_cutoff]
                if(len(halo_candidates)==0):
                    print('| No halos found')
                    print('| Tracking stopped at aexp={0}'.format(aexp_curr))
                    break
                # Filter particles around the selected halo
                sim_curr = sim_curr.d[pynbody.filt.Sphere(p.rvir_search*r200_curr+2*r200_start,[x[k],y[k],z[k]])]
                # Gather particles in the previous selected halo
                halo_part_prev = tree_part_prev.query_radius([x[k],y[k],z[k]],p.rvir*r200_start)[0]
                # Build tree for particles
                tree_part_curr = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
                #print '|    | Previous halo population fractions --------------------'
                print('|    | Previous halo population = {0:7d} --------------------'.format(len(halo_candidates)))
                print('|    | Cutoff mass              = {0:4.2e} -------------------'.format(mass_cutoff*to_msol))
                ids_frac = np.array([])
                # Looping over candidates
                ids_frac = np.zeros(len(halo_candidates))
                ii = 0
                for halo in halo_candidates:

                    # Compute R200 in code units
                    # M200 = 4/3*pi*200*rho_mean*R200^3
                    # In code units, rho_mean=1
                    r200_candidate = (hl[halo,10]*3./(200.*4.*math.pi))**(1.0/3.0)
                    # Gather particle of the halo to track
                    halo_part_curr = tree_part_curr.query_radius(hl[halo,4:7],p.rvir*r200_candidate)[0]
                    # Match unique indices
                    matching_ids = np.where(np.in1d(sim_curr['iord'][halo_part_curr],sim_prev['iord'][halo_part_prev]))[0]
                    # Matching indices fraction
                    ids_frac[ii] = float(len(matching_ids))/float(len(halo_part_prev))
                    ii += 1
                    print('|    |         halo {0:7d} | idf={1:5.2f}% | m={2:5.2e} Msol'.format(halo,100*ids_frac[ii-1],hl[halo,10]*to_msol))
                # Selecting best candidate
                best_candidate = np.argmax(ids_frac)
                id = halo_candidates[best_candidate]
                halo_rejected = np.delete(halo_candidates,best_candidate)
                # Computing Virial radius of the best candidate
                r200_curr = (hl[id,10]*3./(200.*4.*math.pi))**(1.0/3.0)
                mass_curr = hl[id,10]
                print('|    |    -->  halo {0:7d} selected'.format(id))
                print('| ------------------------------------------------------------')

            # Final snapshot - starting point
            else:
                print('| Closest halo coordinates = [{0:.4f},{1:.4f},{2:.4f}] cu'.format(hl[id,4],hl[id,5],hl[id,6]))
                print('| Relative distance        = {0:.2e} cu'.format(np.min(dist_halo)))
                # Loading first snapshot
                sim_curr = _load_sim(list[j-1])
                aexp_curr = float(sim_curr.properties['a'])
                # Computing virial radius
                r200_start = pynbody.analysis.halo.virial_radius(sim_curr.d,cen=hl[id,4:7],r_max=0.5)
                r200_curr = r200_start
                mass_curr = hl[id,10]
                # Code to physical units
                to_mpc = sim_curr.properties['boxsize'].in_units('Mpc')*sim_curr.properties['h']
                to_kpc = 1e3*to_mpc
                to_msol = float(np.sum(sim_curr.d['mass'].in_units('Msol')))
                print('| R200                     = {0:.4f} kpc'.format(r200_start*to_kpc))
                print('| M200                     = {0:.2e} Msol'.format(hl[id,10]*to_msol))
                sim_curr = sim_curr.d[pynbody.filt.Sphere(10*r200_start,hl[id,4:7])]
                sim_curr = sim_curr[np.argsort(sim_curr['iord'])]
                tree_part_curr = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
                print('| ------------------------------------------------------------')
                ids_frac = 1.0
                # Find halos matching coordinate filter around previous halo
                halo_candidates = tree_halo.query_radius(np.squeeze(hl[id,4:7]),p.rvir_search*r200_curr)[0]
                # Filter low mass halos
                halo_candidates = halo_candidates[hl[halo_candidates,10]*to_msol>p.halo_cutoff]
                # Selected halo
                best_candidate = np.where(halo_candidates==id)[0]
                halo_rejected = np.delete(halo_candidates,best_candidate)


            if((np.max(ids_frac)>0.01)&(aexp_curr>p.aexp_min)):
                x[j-1] = hl[id,4]
                y[j-1] = hl[id,5]
                z[j-1] = hl[id,6]
                m[j-1] = hl[id,10]*to_msol
                n[j-1] = hl[id,3]
                if(len(halo_rejected)>0):
                    mnt[j-1] = np.sum(hl[halo_rejected,10])*to_msol
                    mnm[j-1] = np.max(hl[halo_rejected,10])*to_msol
                else:
                    mnt[j-1] = 0.0
                    mnm[j-1] = 0.0
                    idf[j-1] = np.max(ids_frac)
                aexp[j-1] = aexp_curr
                k = j-1
            else:
                print('| Tracking stopped at aexp={0}'.format(aexp_curr))
                break

    # Remove NaNs
    defined = np.where(aexp>0.0)
    x = x[defined]
    y = y[defined]
    z = z[defined]
    m = m[defined]
    n = n[defined]
    mnm = mnm[defined]
    mnt = mnt[defined]
    aexp = aexp[defined]
    idf = idf[defined]

    # Write results
    np.savetxt(p.fname,np.transpose(np.squeeze([aexp,x,y,z,m,n,idf,mnt,mnm])),header="aexp x y z mass npart ids_fraction mass_neighb_max mass_neighb_tot")

    # Fit coefficients
    cx = polyfit(aexp, x, 3, full=True, w=m)[0]
    cy = polyfit(aexp, y, 3, full=True, w=m)[0]
    cz = polyfit(aexp, z, 3, full=True, w=m)[0]

    # Print result
    print('| ------------------------------------------------------------')
    print('| RAMSES polynomial coefficients for camera halo tracking')
    print('| ------------------------------------------------------------')
    print('| xcentre_frame='+','.join('{:6f}'.format(i) for i in cx))
    print('| ycentre_frame='+','.join('{:6f}'.format(i) for i in cy))
    print('| zcentre_frame='+','.join('{:6f}'.format(i) for i in cz))

    # Plotting
    if p.plot:
        flatui = ["#9b59b6", "#3498db", "#95a5a6"]
        cp = sns.color_palette(flatui)
        sns.set_context('poster')
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})

        # Plotting tracked coordinates and fitted polynome
        fig,ax = pyplot.subplots(1)
        ax.plot(aexp, x, 'o', c=cp[0], ms=5)
        ax.plot(aexp, cx[0]+cx[1]*aexp+cx[2]*aexp**2+cx[3]*aexp**3, c=cp[0], lw=3, label='x')
        ax.plot(aexp, y, 'o', c=cp[1], ms=5)
        ax.plot(aexp, cy[0]+cy[1]*aexp+cy[2]*aexp**2+cy[3]*aexp**3, c=cp[1], lw=3, label='y')
        ax.plot(aexp, z, 'o', c=cp[2], ms=5)
        ax.plot(aexp, cz[0]+cz[1]*aexp+cz[2]*aexp**2+cz[3]*aexp**3, c=cp[2], lw=3, label='z')
        ax.set_xlabel('aexp')
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.legend()
        pyplot.savefig(p.fname+".pdf")
        pyplot.close(fig)

        # Plotting mass evolution
        fig,ax = pyplot.subplots(1)
        ax.plot(aexp, np.log10(m), '-', c=cp[0],label='tracked halo')
        ax.plot(aexp, np.log10(mnm), '-', c=cp[1], label='heaviest companion')
        ax.plot(aexp, np.log10(mnt), '-', c=cp[2], label='total companion')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlim([0.0,1.0])
        ax.legend()
        ax.set_xlabel('aexp')
        ax.set_ylabel(r'Mass [M$_{\odot}$]')
        pyplot.savefig(p.fname+"_mass.pdf")
        pyplot.close(fig)
