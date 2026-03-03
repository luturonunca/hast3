import gc
import os
import re
import sys
import glob
import math
import copy
import yt
import warnings
import matplotlib
import numpy as np
import configparser
import seaborn as sns
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore")

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
        try:
            self.full_analysis = config.getboolean('selection','full_analysis')
        except:
            self.full_analysis = False


# ---------------------------------------------------------------------
# yt adapter -- replaces pynbody._load_sim
# Exposes the same interface that select() and its helpers expect:
#   sim['iord'], sim['pos'], sim['x'], sim['y'], sim['z'],
#   sim['mass'].in_units('Msol'), sim.d['mass'],
#   sim.properties['a'], ['h'], ['omegaM0'], ['omegaL0'],
#   sim.properties['boxsize'].in_units('kpc'/'Mpc'),
#   len(sim), sim[integer_array]
# ---------------------------------------------------------------------

def _find_ramses_info(path):
    if path.endswith('.txt') and 'info_' in path:
        return path
    info = glob.glob(path.rstrip('/') + '/info_*.txt')
    if len(info) > 0:
        return sorted(info)[-1]
    return None

class _UnitArray(np.ndarray):
    """ndarray with an in_units() method matching pynbody's SimArray."""
    def __new__(cls, data, units, ds):
        obj = np.asarray(data).view(cls)
        obj._units = units
        obj._ds    = ds
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self._units = getattr(obj, '_units', '')
        self._ds    = getattr(obj, '_ds',    None)
    def in_units(self, target):
        if self._ds is None:
            return np.array(self)
        src = self._units.replace('Msol', 'Msun')
        tgt = target.replace('Msol', 'Msun')
        return np.array(self._ds.arr(np.array(self), src).to(tgt))

class _BoxSize:
    """Mimics pynbody's boxsize unit object."""
    def __init__(self, value_kpc, ds):
        self._value_kpc = value_kpc
        self._ds        = ds
    def in_units(self, target):
        return float(self._ds.quan(self._value_kpc, 'kpc').to(target))

class _YtSimWrapper:
    """Wraps a yt dataset so it looks like a pynbody SimSnap to select()."""
    def __init__(self, pos_kpc, mass_msol, iord, aexp, h, omegaM, omegaL, boxsize_kpc, ds):
        self._pos  = pos_kpc    # (N,3) physical kpc
        self._mass = mass_msol  # (N,)  Msol
        self._iord = iord       # (N,)  int64
        self._ds   = ds
        self.properties = {
            'a':       aexp,
            'h':       h,
            'omegaM0': omegaM,
            'omegaL0': omegaL,
            'boxsize': _BoxSize(boxsize_kpc, ds),
        }
        self.d = self  # pynbody .d namespace (DM subset == all for DM-only sims)

    def __len__(self):
        return len(self._iord)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_field(key)
        # integer-array or slice indexing
        sub = object.__new__(_YtSimWrapper)
        sub._pos  = self._pos[key]
        sub._mass = self._mass[key]
        sub._iord = self._iord[key]
        sub._ds   = self._ds
        sub.properties = self.properties
        sub.d     = sub
        return sub

    def _get_field(self, key):
        if key == 'pos':
            return _UnitArray(self._pos,       'kpc',  self._ds)
        if key == 'x':
            return _UnitArray(self._pos[:, 0], 'kpc',  self._ds)
        if key == 'y':
            return _UnitArray(self._pos[:, 1], 'kpc',  self._ds)
        if key == 'z':
            return _UnitArray(self._pos[:, 2], 'kpc',  self._ds)
        if key == 'mass':
            return _UnitArray(self._mass,      'Msol', self._ds)
        if key == 'iord':
            return self._iord
        raise KeyError(key)


def _load_sim(path):
    info = _find_ramses_info(path)
    if info is None:
        raise IOError('No info_*.txt found in {0}'.format(path))
    ds = yt.load(info)
    ad = ds.all_data()
    try:
        pos_x = ad[('DM', 'particle_position_x')].to('kpc').value
        pos_y = ad[('DM', 'particle_position_y')].to('kpc').value
        pos_z = ad[('DM', 'particle_position_z')].to('kpc').value
        mass  = ad[('DM', 'particle_mass')].to('Msun').value
        iord  = np.array(ad[('DM', 'particle_identity')]).astype(np.int64)
    except Exception:
        pos_x = ad[('all', 'particle_position_x')].to('kpc').value
        pos_y = ad[('all', 'particle_position_y')].to('kpc').value
        pos_z = ad[('all', 'particle_position_z')].to('kpc').value
        mass  = ad[('all', 'particle_mass')].to('Msun').value
        iord  = np.array(ad[('all', 'particle_identity')]).astype(np.int64)
    pos = np.vstack((pos_x, pos_y, pos_z)).T
    boxsize_kpc = float(ds.domain_width[0].to('kpc'))
    try:    aexp   = float(ds.scale_factor)
    except: aexp   = 1.0
    try:    h      = float(ds.hubble_constant)
    except: h      = 0.7
    try:    omegaM = float(ds.omega_matter)
    except: omegaM = 0.3
    try:    omegaL = float(ds.omega_lambda)
    except: omegaL = 0.7
    return _YtSimWrapper(pos, mass, iord, aexp, h, omegaM, omegaL, boxsize_kpc, ds)


# ---------------------------------------------------------------------
# virial radius -- replaces pynbody.analysis.halo.virial_radius
# Same signature: virial_radius(sim, cen, r_max)
# Returns radius in the same units as sim positions (physical kpc).
# ---------------------------------------------------------------------

def _virial_radius(pos, mass, boxsize_kpc, cen, r_max):
    center = np.array(cen)
    mean_density = np.sum(mass) / (boxsize_kpc**3)
    r = np.linalg.norm(pos - center, axis=1)
    idx = np.where(r <= r_max)[0]
    if idx.size == 0:
        return 0.0
    r_in = r[idx]
    m_in = mass[idx]
    order    = np.argsort(r_in)
    r_sorted = r_in[order]
    cum_mass = np.cumsum(m_in[order])
    valid    = r_sorted > 0.0
    r_sorted = r_sorted[valid]
    cum_mass = cum_mass[valid]
    if r_sorted.size == 0:
        return 0.0
    density = cum_mass / ((4.0/3.0) * math.pi * r_sorted**3)
    ok = np.where(density >= 200.0 * mean_density)[0]
    if ok.size == 0:
        return 0.0
    return float(r_sorted[ok[-1]])


# ---------------------------------------------------------------------
# everything below is verbatim from hast_pynbody.py
# ---------------------------------------------------------------------

def _clump_header_format(path):
    try:
        with open(path, 'r') as f:
            header = f.readline().strip()
        header = header.lstrip('#').strip().lower()
        if "peak_x" in header and "mass_cl" in header:
            return "new"
    except Exception:
        pass
    return "legacy"

def _normalize_clump_columns(data, fmt):
    if fmt != "new":
        return data
    # Current format: index, lev, parent, ncell, peak_x, peak_y, peak_z,
    # rho-, rho+, rho_av, mass_cl, relevance -- positions and mass are
    # already in the slots the rest of the code expects (4:7 and 10).
    return data

def halo_list(output,quiet=False,clump_mass_unit='fraction'):

    list = glob.glob(output+'/clump_?????.txt?????')
    if(not quiet):
        print('| ------------------------------------------------------------')
        print('| Reading RAMSES clump finder files')
        print('| ------------------------------------------------------------')
        print('| nfiles        = {0}'.format(len(list)))
    fmt = _clump_header_format(list[0]) if len(list) > 0 else "legacy"
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
    data_all = _normalize_clump_columns(data_all, fmt)
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


def halo_list_yt(sim, halo_finder, quiet=False):
    if not quiet:
        print('| ------------------------------------------------------------')
        print('| Running {0} halo finder (yt)'.format(halo_finder))
        print('| ------------------------------------------------------------')
    try:
        from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
    except Exception:
        print('[Error] yt halo_analysis not available; install yt_astro_analysis')
        sys.exit()
    try:
        hc = HaloCatalog(data_ds=sim._ds, finder_method=halo_finder)
        hc.create()
    except Exception as e:
        print('[Error] halo finder "{0}" failed: {1}'.format(halo_finder, e))
        sys.exit()
    try:
        halos = hc.halos
        n_halos = len(halos)
    except Exception:
        print('[Error] could not read halo catalog')
        sys.exit()
    if not quiet:
        print('| nhalos        = {0}'.format(n_halos))
    data = np.zeros((n_halos, 11), dtype=float)
    boxsize_kpc = float(sim.properties['boxsize'].in_units('kpc'))
    for i in range(n_halos):
        data[i, 0]   = i
        data[i, 4:7] = _halo_center_to_unit_box(sim, halos[i])
        data[i, 10]  = _halo_mass_msol(halos[i])
    if np.max(data[:, 4:7]) <= 1.0:
        data[:, 4:7] *= boxsize_kpc
    data_sorted = data[data[:, 10].argsort()]
    return data_sorted


def plot_candidates(data,sim,center=[0.,0.,0.],comoving=False):
    sns.set_context('poster')
    sns.set_style("ticks",{"axes.grid": False,"xtick.direction":'in',"ytick.direction":'in'})
    cp2 = sns.color_palette("husl",len(data[:,0]))
    print('| Plotting ',len(data[:,0]),' haloes')
    fig,ax = pyplot.subplots(1,2,figsize=(18,8),sharex=True)
    proj =[['y','x'],['z','x']]
    dproj =[[5,4],[6,4]]
    if comoving:
        try:
            aexp = float(sim.properties['a'])
        except Exception:
            aexp = 1.0
        data_plot = data.copy()
        data_plot[:,4:7] /= aexp
    else:
        data_plot = data
    for i in range(len(ax)):
        x=proj[i][0]
        y=proj[i][1]
        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)
        if comoving:
            sim_x = sim.d[x] / aexp
            sim_y = sim.d[y] / aexp
        else:
            sim_x = sim.d[x]
            sim_y = sim.d[y]
        if np.max(sim_x) > 1.0 or np.max(sim_y) > 1.0:
            try:
                boxsize = float(sim.properties['boxsize'].in_units('kpc'))
                if comoving:
                    boxsize /= aexp
            except Exception:
                boxsize = float(np.max([np.max(sim_x), np.max(sim_y)]))
            hist_range = [[0.0, boxsize], [0.0, boxsize]]
        else:
            hist_range = [[0.0, 1.0], [0.0, 1.0]]
        im,xedges,yedges = np.histogram2d(
            sim_x, sim_y, weights=sim.d['mass'], bins=512, range=hist_range)
        im = np.rot90(im)
        b = ax[i].get_position()
        data_plot[:,4:7] -= center
        h = ax[i].scatter(data_plot[:,dproj[i][0]],data_plot[:,dproj[i][1]],s=50,c=cp2,alpha=0.5)
        ax[i].set(adjustable='box', aspect='equal')
        extent_max = hist_range[0][1]
        tv = ax[i].imshow(
            np.log10(im), cmap='bone_r', interpolation='quadric',
            aspect='equal', extent=[0.0, extent_max, 0.0, extent_max])
        ax[i].set_xlim([0.0-center[0], extent_max-center[0]])
        ax[i].set_ylim([0.0-center[1], extent_max-center[1]])
        for j in range(len(data_plot[:,0])):
            ax[i].annotate(str(j+1),(data_plot[j,dproj[i][0]]+0.01,data_plot[j,dproj[i][1]]+0.01),color=cp2[j])

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
        d = halo_list_yt(sim_zlast, p.halo_finder)
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

    print('| Halo position range')
    print('|   x [{0:.2f}, {1:.2f}] kpc'.format(np.min(d[:,4]), np.max(d[:,4])))
    print('|   y [{0:.2f}, {1:.2f}] kpc'.format(np.min(d[:,5]), np.max(d[:,5])))
    print('|   z [{0:.2f}, {1:.2f}] kpc'.format(np.min(d[:,6]), np.max(d[:,6])))
    print('| Boundary [{0:.2f}, {1:.2f}] kpc'.format(boundary_min, boundary_max))
    print('| Box size  {0:.2f} kpc'.format(float(sim_zlast.properties['boxsize'].in_units('kpc'))))
    print('| ------------------------------------------------------------')

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
        if((d[candidates[0][i],4]<boundary_min)or(d[candidates[0][i],4]>boundary_max)):
            flag[i] = 4
        if((d[candidates[0][i],5]<boundary_min)or(d[candidates[0][i],5]>boundary_max)):
            flag[i] = 4
        if((d[candidates[0][i],6]<boundary_min)or(d[candidates[0][i],6]>boundary_max)):
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
        halo_colors = sns.color_palette("husl",wh1[0].size)
        if(p.plot):
            cp = halo_colors
            ax=plot_candidates(d[candidates[0][wh1],:],sim_zlast,comoving=True)
            if(not p.plot_traceback):
                pyplot.savefig(p.fname+'.pdf',dpi=100)
            print('| ------------------------------------------------------------')
        elif(p.full_analysis):
            cp = halo_colors
            ax=plot_candidates(d[candidates[0][wh1],:],sim_zlast,comoving=True)
            print('| ------------------------------------------------------------')
        hull_vols = []
        hull_dens_vals = []
        hull_halo_idx = []
        print('| Building Tree [{0} particles]'.format(len(sim_zlast)))
        print('| First halo pos  : {0}'.format(d[candidates[0][wh1[0][0]], 4:7]))
        print('| First part pos  : {0}'.format(sim_zlast['pos'][0]))
        tree = KDTree(np.squeeze((sim_zlast['pos'])),leaf_size=p.tree_nleaves)
        r200 = np.array([])
        _part_pos     = np.array(sim_zlast['pos'])
        _part_mass    = np.array(sim_zlast['mass'].in_units('Msol'))
        _boxsize_kpc  = float(sim_zlast.properties['boxsize'].in_units('kpc'))
        print('| Computing Virial radii')
        for i in range(wh1[0].size):
            try:
                rr = _virial_radius(_part_pos,_part_mass,_boxsize_kpc,cen=d[candidates[0][wh1[0][i]],4:7],r_max=rbuffer)
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
            if npart == 0:
                print('|     | --- No particles traced back; skipping')
                print('| ------------------------------------------------------------')
                continue
            safety = False
            try:
                box_kpc = float(sim_zinit.properties['boxsize'].in_units('kpc'))
            except Exception:
                box_kpc = 1.0
            if((np.max(sim_zinit['x'][region_zinit])-np.min(sim_zinit['x'][region_zinit]))/box_kpc>0.5):
                safety = True
            if((np.max(sim_zinit['y'][region_zinit])-np.min(sim_zinit['y'][region_zinit]))/box_kpc>0.5):
                safety = True
            if((np.max(sim_zinit['z'][region_zinit])-np.min(sim_zinit['z'][region_zinit]))/box_kpc>0.5):
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
            if m200 < p.min_mass:
                print('|     | --- m200 {0:.2e} Msol < min_mass {1:.2e} Msol; skipping'.format(m200,p.min_mass))
                print('| ------------------------------------------------------------')
                continue
            lambda200 = 0.0
            xmean = float(np.mean(sim_zinit['x'][region_zinit]))
            ymean = float(np.mean(sim_zinit['y'][region_zinit]))
            zmean = float(np.mean(sim_zinit['z'][region_zinit]))
            print('|     | --- Candidate halo properties')
            print('|     | --------------- m200                   -> {0:.3e} Msol'.format(m200))
            print('|     | --------------- r200                   -> [{0:.1f} kpc phys,{1:.1f} kpc comov, {2:.4f} cu]'.format(r200[i],r200[i]/sim_zlast.properties['a'],r200[i]/to_kpc))
            print('|     | --------------- lambda                 -> {0:.4f}'.format(lambda200))
            print('|     | --------------- npart(r<r200)          -> {0}'.format(npart_r200))
            print('|     | --- Candidate halo position            -> [{0:.5f},{1:.5f},{2:.5f}]'.format(pos_candidate[0],pos_candidate[1],pos_candidate[2]))
            print('|     | --- Mean particle position in ICs      -> [{0:.5f},{1:.5f},{2:.5f}]'.format(xmean,ymean,zmean))
            if npart < 4:
                print('|     | --- Not enough particles for convex hull; skipping')
                print('| ------------------------------------------------------------')
                continue
            hull = ConvexHull(sim_zinit['pos'][region_zinit]-sim_zinit['pos'][region_zinit].mean(axis=0))
            if(p.full_analysis):
                hull_vols.append(hull.volume)
                hull_dens_vals.append(float(np.sum(sim_zinit['mass'][region_zinit])/hull.volume))
                hull_halo_idx.append(i)
            if((p.plot or p.full_analysis)and(p.plot_traceback)):
                proj =[['y','x'],['z','x']]
                dproj =[[5,4],[6,4]]
                for k in range(len(ax)):
                    x=proj[k][0]
                    y=proj[k][1]
                    points_2d = np.squeeze([[
                        sim_zinit[x][region_zinit] / sim_zinit.properties['a']
                    ],[
                        sim_zinit[y][region_zinit] / sim_zinit.properties['a']
                    ]]).transpose()
                    hull2d = ConvexHull(points_2d)
                    aexp_init = sim_zinit.properties['a']
                    xvals = sim_zinit[x][region_zinit] / aexp_init
                    yvals = sim_zinit[y][region_zinit] / aexp_init
                    ax[k].plot(xvals[np.append(hull2d.vertices,hull2d.vertices[0])],yvals[np.append(hull2d.vertices,hull2d.vertices[0])],'k-',lw=2,color=cp[i])
                    left=np.argmin(xvals[hull2d.vertices])

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
        if(p.full_analysis):
            pdf = PdfPages(p.fname+'_analysis.pdf')
            pdf.savefig(ax[0].get_figure(),dpi=100)
            if(len(hull_vols)>0):
                fig_scatter,ax_s = pyplot.subplots(figsize=(8,6))
                for k in range(len(hull_vols)):
                    ax_s.scatter(hull_vols[k],hull_dens_vals[k],
                                 color=halo_colors[hull_halo_idx[k]],s=100,zorder=5)
                    ax_s.annotate(str(hull_halo_idx[k]+1),(hull_vols[k],hull_dens_vals[k]),
                                  textcoords='offset points',xytext=(6,4),
                                  color=halo_colors[hull_halo_idx[k]])
                ax_s.set_xscale('log')
                ax_s.set_yscale('log')
                ax_s.set_xlabel('Lagrangian volume [kpc$^3$]')
                ax_s.set_ylabel('Lagrangian density [M$_\\odot$ kpc$^{-3}$]')
                ax_s.set_title('Lagrangian region: volume vs density')
                sns.despine()
                pdf.savefig(fig_scatter,dpi=100)
                pyplot.close(fig_scatter)
            pdf.close()
            print('| Full analysis saved to {0}_analysis.pdf'.format(p.fname))

    else:
        print('| No haloes matching the criteria')
        return

    return
