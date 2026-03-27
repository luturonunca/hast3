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
from numpy.polynomial.polynomial import polyfit

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
        try:
            self.merger_tree = config.getboolean('selection','merger_tree')
        except:
            self.merger_tree = True
        try:
            self.levelmin = config.getint('selection','levelmin')
        except:
            self.levelmin = 7
        try:
            self.levelmax = config.getint('selection','levelmax')
        except:
            self.levelmax = 11
        try:
            self.padding = config.getint('selection','padding')
        except:
            self.padding = 16
        try:
            self.music_margin = config.getfloat('selection','music_margin')
        except:
            self.music_margin = 0.1
        try:
            self.neighb_mass_frac = config.getfloat('selection','neighb_mass_frac')
        except:
            self.neighb_mass_frac = 0.3


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

def _read_info_params(output_dir):
    """Read aexp, unit_l, unit_d, boxlen, H0, omega_m, omega_l from info_*.txt.
    Fast text read — does not load particle data."""
    files = glob.glob(output_dir + '/info_?????.txt')
    if not files:
        return {}
    params = {}
    want = {'aexp', 'unit_l', 'unit_d', 'boxlen', 'H0', 'omega_m', 'omega_l'}
    with open(files[0], 'r') as f:
        for line in f:
            if '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                if key in want:
                    try:
                        params[key] = float(val.strip())
                    except ValueError:
                        pass
    return params


def _top_halos_kpc_msol(output_dir, params):
    """Read top-level halos from halo_*.txt (falls back to clump_*.txt).
    Returns (N,5) array: [index, x_kpc, y_kpc, z_kpc, mass_msol].
    Returns None if no files are found."""
    _kpc   = 3.085677581e21   # cm per kpc
    _msol  = 1.9885e33        # g per Msol
    unit_l = params.get('unit_l', _kpc)
    unit_d = params.get('unit_d', 1.0)
    boxlen = params.get('boxlen', 1.0)
    to_kpc  = unit_l / _kpc          # 1 code length -> kpc
    to_msol = unit_d * unit_l**3 / _msol   # 1 code mass -> Msol

    files = glob.glob(output_dir + '/halo_?????.txt?????')
    use_halo = len(files) > 0
    if not use_halo:
        files = glob.glob(output_dir + '/clump_?????.txt?????')
    if not files:
        return None

    chunks = []
    for f in files:
        try:
            data = np.loadtxt(f, skiprows=1)
        except Exception:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0:
            continue
        chunks.append(data)
    if not chunks:
        return None
    data_all = np.vstack(chunks)

    if use_halo:
        # halo file cols: index(0) ncell(1) x(2) y(3) z(4) rho+(5) mass(6)
        idx = data_all[:, 0]
        pos = data_all[:, 2:5]
        m   = data_all[:, 6]
    else:
        # clump file cols: index(0) lev(1) parent(2) ncell(3) x(4) y(5) z(6) ... mass(10)
        # top-level: parent == index
        top = data_all[:, 2] == data_all[:, 0]
        if not np.any(top):
            return None
        data_all = data_all[top]
        idx = data_all[:, 0]
        pos = data_all[:, 4:7]
        m   = data_all[:, 10]

    # Positions in RAMSES clump/halo files are in [0,1] box-fraction units.
    # unit_l = aexp * box_Mpc * 3.08e24 / (h/100) [cm] = full physical box in cm.
    # So pos * (unit_l/kpc_cm) gives physical kpc directly.
    # NOTE: 'boxlen' in the RAMSES info file is the coarse cell size (1/2^levelmin),
    # NOT the full box length, so it must NOT be used as a multiplier here.
    if np.max(pos) > 1.5:
        # Positions written in cell units [0, nx_loc]; convert to [0,1].
        # boxlen_info = 1/nx_loc, so multiply by boxlen to normalise.
        pos = pos * boxlen
    pos_kpc = pos * to_kpc
    m_msol  = m * to_msol

    return np.column_stack([idx, pos_kpc, m_msol])


def _top_halos_box_msol(output_dir, params):
    """Read top-level halos. Returns (N,5) array: [index, x, y, z, mass_msol].
    Positions are in [0,1] box-fraction units (no kpc conversion).
    Mass in Msol."""
    _msol   = 1.9885e33
    unit_d  = params.get('unit_d', 1.0)
    unit_l  = params.get('unit_l', 3.085677581e21)
    boxlen  = params.get('boxlen', 1.0)
    to_msol = unit_d * unit_l**3 / _msol

    files = glob.glob(output_dir + '/halo_?????.txt?????')
    use_halo = len(files) > 0
    if not use_halo:
        files = glob.glob(output_dir + '/clump_?????.txt?????')
    if not files:
        return None

    chunks = []
    for f in files:
        try:
            data = np.loadtxt(f, skiprows=1)
        except Exception:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0:
            continue
        chunks.append(data)
    if not chunks:
        return None
    data_all = np.vstack(chunks)

    if use_halo:
        idx = data_all[:, 0]
        pos = data_all[:, 2:5]
        m   = data_all[:, 6]
    else:
        top = data_all[:, 2] == data_all[:, 0]
        if not np.any(top):
            return None
        data_all = data_all[top]
        idx = data_all[:, 0]
        pos = data_all[:, 4:7]
        m   = data_all[:, 10]

    # If positions are in cell units [0, nx_loc], normalise to [0, 1]
    if np.max(pos) > 1.5:
        pos = pos * boxlen

    return np.column_stack([idx, pos, m * to_msol])


def _read_all_halos_box(output_dir, params):
    """Read ALL halos (including subhalos) from clump files.
    Returns (N,6) array: [idx, x, y, z, mass_msol, parent_idx].
    Positions in [0,1] box fractions. Returns None if no files found."""
    _msol   = 1.9885e33
    unit_d  = params.get('unit_d', 1.0)
    unit_l  = params.get('unit_l', 3.085677581e21)
    boxlen  = params.get('boxlen', 1.0)
    to_msol = unit_d * unit_l**3 / _msol

    files = glob.glob(output_dir + '/clump_?????.txt?????')
    if not files:
        return None
    chunks = []
    for f in files:
        try:
            data = np.loadtxt(f, skiprows=1)
        except Exception:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.size == 0:
            continue
        chunks.append(data)
    if not chunks:
        return None
    data_all = np.vstack(chunks)
    # clump cols: index(0) lev(1) parent(2) ncell(3) x(4) y(5) z(6) ... mass(10)
    idx    = data_all[:, 0]
    parent = data_all[:, 2]
    pos    = data_all[:, 4:7]
    m      = data_all[:, 10] * to_msol
    if np.max(pos) > 1.5:
        pos = pos * boxlen
    return np.column_stack([idx, pos, m, parent])  # (N,6)


def _read_npart_threshold(output_dir, default=20):
    """Read mass_threshold from namelist.txt in the snapshot directory.
    In CLUMPFIND_PARAMS, mass_threshold is in units of the minimum particle
    mass, so it is effectively a minimum particle count."""
    path = os.path.join(output_dir, 'namelist.txt')
    if not os.path.exists(path):
        return default
    in_block = False
    with open(path, 'r') as f:
        for line in f:
            l = line.strip().lower()
            if l.startswith('&clumpfind_params'):
                in_block = True
                continue
            if in_block:
                if l.startswith('/') or l.startswith('&'):
                    break
                if 'mass_threshold' in l and '=' in l:
                    try:
                        return int(float(l.split('=')[1].strip().split()[0].rstrip(',')))
                    except (ValueError, IndexError):
                        pass
    return default


def _r200_kpc(mass_msol, params):
    """Estimate R200 in kpc from halo mass in Msol using critical density at snapshot z."""
    H0   = params.get('H0',      70.0)
    aexp = params.get('aexp',     1.0)
    om   = params.get('omega_m',  0.3)
    ol   = params.get('omega_l',  0.7)
    z    = 1.0 / aexp - 1.0
    Hz   = H0 * np.sqrt(om * (1.0 + z)**3 + ol)  # km/s/Mpc
    Hz_si = Hz * 1e3 / 3.085677581e22             # 1/s
    G_si  = 6.674e-11                             # m^3 kg^-1 s^-2
    rho_crit_si  = 3.0 * Hz_si**2 / (8.0 * np.pi * G_si)  # kg/m^3
    rho_crit_cgs = rho_crit_si * 1e-3                     # g/cm^3  (1 kg/m^3 = 1e-3 g/cm^3)
    _kpc  = 3.085677581e21
    _msol = 1.9885e33
    rho_crit_msol_kpc3 = rho_crit_cgs * _kpc**3 / _msol
    return (3.0 * mass_msol / (4.0 * np.pi * 200.0 * rho_crit_msol_kpc3))**(1.0 / 3.0)


def _lookback_gyr(z_vals, params):
    """Lookback time in Gyr for scalar or array z_vals (flat ΛCDM)."""
    H0 = params.get('H0',     70.0)   # km/s/Mpc
    om = params.get('omega_m', 0.3)
    ol = params.get('omega_l', 0.7)
    H0_gyr = H0 * 1.022e-3            # Gyr^-1
    scalar = np.isscalar(z_vals)
    zv = np.atleast_1d(np.asarray(z_vals, dtype=float))
    result = np.zeros(len(zv))
    for i, z in enumerate(zv):
        if z <= 0.0:
            result[i] = 0.0
            continue
        zz = np.linspace(0.0, z, 500)
        intgd = 1.0 / ((1.0 + zz) * np.sqrt(om * (1.0 + zz)**3 + ol))
        result[i] = np.trapz(intgd, zz) / H0_gyr
    return float(result[0]) if scalar else result


def _save_merger_tree(fname, halo_id, nodes, edges):
    """Persist merger tree nodes and edges to a JSON cache file."""
    import json
    payload = {
        'halo_id': halo_id,
        'nodes': [
            {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in nd.items()}
            for nd in nodes
        ],
        'edges': [list(e) for e in edges],
    }
    path = '{0}_tree_{1}.json'.format(fname, halo_id)
    with open(path, 'w') as f:
        json.dump(payload, f)
    print('| Merger tree cached to {0}'.format(path))


def _load_merger_tree(fname, halo_id):
    """Load merger tree from cache. Returns (nodes, edges) or (None, None)."""
    import json
    path = '{0}_tree_{1}.json'.format(fname, halo_id)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        nodes = []
        for nd in payload['nodes']:
            nd['pos']  = np.array(nd['pos'])
            if 'iord' in nd:
                nd['iord'] = np.array(nd['iord'], dtype=np.int64)
            nodes.append(nd)
        edges = [tuple(e) for e in payload['edges']]
        print('| Merger tree loaded from cache: {0}'.format(path))
        return nodes, edges
    except Exception as e:
        print('| [Warning] Could not read merger tree cache {0}: {1}'.format(path, e))
        return None, None


def _lookback_time_gyr(aexp, params):
    """Lookback time in Gyr at a given aexp, using flat LCDM from info params."""
    H0 = params.get('H0', 70.0)
    om = params.get('omega_m', 0.3)
    ol = params.get('omega_l', 0.7)
    H0s = H0 * 1e3 / 3.085677581e22          # 1/s
    tH  = 1.0 / H0s / 3.15576e16             # Gyr
    def _t(a):
        return (2.0 / (3.0 * np.sqrt(ol))) * np.arcsinh(np.sqrt(ol / om) * a**1.5) * tH
    return _t(1.0) - _t(aexp)                 # lookback = t0 - t(z)


def _add_redshift_top_axis(ax, params, z_ticks=(0, 0.5, 1, 2, 3, 5, 7, 10)):
    """Add a secondary x-axis on top showing redshift values."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    t_ticks, z_labels = [], []
    for z in z_ticks:
        a = 1.0 / (1.0 + z)
        t = _lookback_time_gyr(a, params)
        x0, x1 = ax.get_xlim()
        if x0 <= t <= x1:
            t_ticks.append(t)
            z_labels.append(str(z))
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(z_labels, fontsize=8)
    ax2.set_xlabel('Redshift', fontsize=9)
    return ax2


def plot_merger_tree(nodes, edges, ax_tree, ax_mass, params,
                     halo_label='', z_scale=None, proj=('x', 'y')):
    """Scatter DM particles of each tree node, y-shifted by redshift.

    Particles belonging to each node are scattered in the proj[0]-proj[1]
    plane (kpc). The proj[1] coordinate is shifted by z/(1+z) * z_scale so that
    z=0 nodes are unshifted and higher-redshift snapshots appear progressively
    offset upward. The z/(1+z) mapping gives more visual separation at low
    redshift where snapshots are closely spaced in z.
    Horizontal dotted lines label each snapshot's redshift.
    Main branch is blue, merger branches are orange.

    ax_mass is hidden (kept for pipeline compatibility).
    z_scale : kpc per unit z/(1+z). Defaults to 0.1 * box_kpc of root snapshot.
    proj    : spatial axes to project, chosen from 'x', 'y', 'z'.
    """
    if not nodes:
        ax_tree.text(0.5, 0.5, 'No tree data',
                     transform=ax_tree.transAxes, ha='center')
        return

    col_main   = '#3182bd'
    col_merger = '#e6550d'
    axis_map   = {'x': 0, 'y': 1, 'z': 2}
    xi = axis_map[proj[0]]
    yi = axis_map[proj[1]]

    # Load each unique snapshot once; positions in kpc
    sim_cache = {}
    for nd in nodes:
        snap = nd['snap']
        if snap in sim_cache:
            continue
        try:
            sim     = _load_sim(snap)
            box_kpc = float(sim.properties['boxsize'].in_units('kpc'))
            sim_cache[snap] = (np.array(sim['pos']), sim['iord'], box_kpc)
        except Exception as e:
            print('[plot_merger_tree] could not load {0}: {1}'.format(snap, e))

    print('[plot_merger_tree] {0} nodes, {1} snapshots loaded'.format(
        len(nodes), len(sim_cache)))

    # Default z_scale: 0.1 box width per unit redshift
    if z_scale is None:
        snap0 = nodes[0]['snap']
        box_kpc = sim_cache[snap0][2] if snap0 in sim_cache else 100000.0
        z_scale = box_kpc * 0.1

    # Pre-compute main-branch centre per snapshot for consistent centering
    # All branches at the same snapshot are centred on the main branch mean,
    # so merger branches appear spatially offset rather than collapsed to zero.
    main_centre = {}   # snap -> (x_mean_kpc, y_mean_kpc)
    for nd in nodes:
        if not nd['is_main']:
            continue
        snap = nd['snap']
        if snap not in sim_cache:
            continue
        pos_kpc, iord_snap, box_kpc = sim_cache[snap]
        if 'iord' in nd and len(nd['iord']) > 0:
            mask = np.isin(iord_snap, nd['iord'])
        else:
            dists = np.linalg.norm(pos_kpc - np.array(nd['pos']), axis=1)
            mask  = dists <= nd['r200']
        if np.any(mask):
            main_centre[snap] = (float(np.mean(pos_kpc[mask, xi])),
                                 float(np.mean(pos_kpc[mask, yi])))

    # Scatter particles for each node; track centred x range and per-node mean x
    seen_z      = {}   # z (rounded) -> y_shift, for redshift labels
    x_all       = []   # all centred x values, to set xlim from actual spread
    node_xmean  = {}   # nd_idx -> mean centred x of that node's particles
    node_yshift = {}   # nd_idx -> y_shift of that node
    for nd_idx, nd in enumerate(nodes):
        snap = nd['snap']
        if snap not in sim_cache:
            continue
        pos_kpc, iord_snap, box_kpc = sim_cache[snap]

        # Use stored iord if available, else fall back to R200 sphere (both in kpc)
        if 'iord' in nd and len(nd['iord']) > 0:
            mask = np.isin(iord_snap, nd['iord'])
        else:
            dists = np.linalg.norm(pos_kpc - np.array(nd['pos']), axis=1)
            mask  = dists <= nd['r200']

        if not np.any(mask):
            print('[plot_merger_tree] node halo_id={0} z={1:.2f}: no particles found'.format(
                nd['halo_id'], nd['z']))
            continue

        print('[plot_merger_tree] node halo_id={0} z={1:.2f}: {2} particles'.format(
            nd['halo_id'], nd['z'], np.sum(mask)))

        cx, cy  = main_centre.get(snap, (float(np.mean(pos_kpc[mask, xi])),
                                         float(np.mean(pos_kpc[mask, yi]))))
        x_cent  = pos_kpc[mask, xi] - cx
        y_cent  = pos_kpc[mask, yi] - cy
        y_shift = (nd['z'] / (1.0 + nd['z'])) * z_scale
        col     = col_main if nd['is_main'] else col_merger
        ax_tree.scatter(x_cent, y_cent + y_shift,
                        s=0.5, color=col, alpha=0.4, rasterized=True)

        x_all.append(x_cent)
        node_xmean[nd_idx]  = float(np.mean(x_cent))
        node_yshift[nd_idx] = y_shift
        z_key = round(nd['z'], 3)
        if z_key not in seen_z:
            seen_z[z_key] = y_shift

    # xlim from actual particle spread
    if x_all:
        x_concat   = np.concatenate(x_all)
        half_width = np.max(np.abs(x_concat)) * 3.0
        ax_tree.set_xlim(-half_width, half_width)
    else:
        half_width = 1.0

    # Horizontal dotted lines; label only the snapshot nearest each integer redshift
    int_z_targets = list(range(int(np.floor(max(seen_z.keys()))) + 1))  # [0,1,2,...]
    labelled = set()
    for z_int in int_z_targets:
        if not seen_z:
            break
        closest = min(seen_z.keys(), key=lambda z: abs(z - z_int))
        labelled.add(closest)
    for z_val, y_sh in sorted(seen_z.items()):
        ax_tree.axhline(y_sh, color='grey', lw=0.5, ls=':', alpha=0.6, zorder=0)
        if z_val in labelled:
            ax_tree.text(half_width, y_sh, ' z={0:.1f}'.format(z_val),
                         va='center', ha='left', fontsize=12, color='grey')

    # Mass ratio annotations at merger events
    # For each merger edge, find the corresponding main node at the same snapshot/descendant
    # Build a lookup: (desc_idx, snap) -> main node_idx
    main_at = {}   # (desc_idx, snap) -> node_idx of the main progenitor
    for (nidx, didx, etype) in edges:
        if etype == 'main':
            main_at[(didx, nodes[nidx]['snap'])] = nidx
    for (nidx, didx, etype) in edges:
        if etype != 'merger':
            continue
        if nidx not in node_xmean:
            continue
        main_nidx = main_at.get((didx, nodes[nidx]['snap']))
        if main_nidx is None:
            continue
        m_merger = nodes[nidx]['mass']
        m_main   = nodes[main_nidx]['mass']
        if m_main <= 0:
            continue
        ratio_n = m_main / m_merger if m_merger > 0 else 0.0
        label   = '{0:.0f}:1'.format(ratio_n)
        ax_tree.text(-half_width * 0.97, node_yshift[nidx], label,
                     va='center', ha='left', fontsize=8, color=col_merger)

    ax_tree.set_xlabel('')
    ax_tree.set_ylabel('')
    ax_tree.set_title('Merger tree — halo {0}'.format(halo_label))
    ax_tree.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_tree.tick_params(axis='y', which='both', left=False, labelleft=False)
    sns.despine(ax=ax_tree)

    # --- Mass evolution plot ---
    # Build desc->main-progenitor map to trace main branch chain backward
    desc_to_main_prog = {}
    for (prog_idx, desc_idx, etype) in edges:
        if etype == 'main':
            desc_to_main_prog[desc_idx] = prog_idx

    def _trace_backward(start):
        chain = [start]
        cur = start
        while cur in desc_to_main_prog:
            cur = desc_to_main_prog[cur]
            chain.append(cur)
        return chain  # start = lowest z, end = highest z

    main_chain = _trace_backward(0) if nodes else []

    if main_chain:
        z_main = np.array([nodes[i]['z'] for i in main_chain])
        t_main = _lookback_gyr(z_main, params)
        m_main = np.array([nodes[i]['mass'] for i in main_chain])

        # Drop outliers: keep only points where mass is within 10x of local median
        if len(m_main) >= 3:
            med  = np.median(m_main)
            keep = (m_main > med / 10.0) & (m_main < med * 10.0)
        else:
            keep = np.ones(len(m_main), dtype=bool)
        ax_mass.plot(t_main[keep], m_main[keep], color=col_main, lw=2)

    # Vertical orange line at the lookback time of each merger event
    merger_times = []
    for (prog_idx, desc_idx, etype) in edges:
        if etype == 'merger':
            merger_times.append(_lookback_gyr(nodes[prog_idx]['z'], params))
    for t_mrg in merger_times:
        ax_mass.axvline(t_mrg, color=col_merger, lw=1, alpha=0.7)

    ax_mass.set_yscale('log')
    # today (z=0, t=0) on the LEFT; lookback time increases to the right
    ax_mass.set_xlabel('Lookback time [Gyr]')
    ax_mass.set_ylabel('Mass [M$_\\odot$]')

    # Top x-axis: redshift ticks at integer z values, aligned to lookback times
    z_max_plot = max((nodes[i]['z'] for i in main_chain), default=0.0)
    ax_z = ax_mass.twiny()
    ax_z.set_xlim(ax_mass.get_xlim())
    z_int_ticks = list(range(0, int(np.floor(z_max_plot)) + 1))
    t_int_ticks = _lookback_gyr(z_int_ticks, params)
    ax_z.set_xticks(t_int_ticks)
    ax_z.set_xticklabels([str(z) for z in z_int_ticks])
    ax_z.set_xlabel('Redshift')
    sns.despine(ax=ax_mass)


def build_merger_tree(sim_dir, halo_ids, output_zlast, output_zinit,
                      r_search_factor=1.0, z_max=6.0):
    """Build merger trees for one or more halos in a single pass through snapshots.

    halo_ids : int or list of ints
    z_max    : stop traversal at snapshots with z > z_max (default 6.0)
    Returns  : dict { halo_id -> (nodes, edges) }

    At each snapshot (loaded once for all halos):
    1. Find tracked particles by iord matching (position-independent).
    2. Assign each particle to the halo whose R200 contains it.
       If a particle is inside both a host and its subhalo, assign to the
       subhalo (innermost). If inside two peer halos, assign to the nearest
       centre.
    3. Drop halos with fewer than mass_threshold particles (read from
       output_dir/namelist.txt; defaults to 20).
    4. The progenitor with the most particles is the 'main' branch; the
       rest are 'merger' branches, each carrying its own iord subset forward.
    """
    def _snap_num(path):
        return int(os.path.basename(path).split('_')[1])

    if isinstance(halo_ids, (int, np.integer)):
        halo_ids = [int(halo_ids)]
    else:
        halo_ids = [int(h) for h in halo_ids]

    n_zinit = _snap_num(output_zinit)
    n_zlast = _snap_num(output_zlast)
    all_snaps = sorted(glob.glob(os.path.join(sim_dir, 'output_?????')))
    snaps = [s for s in all_snaps if n_zinit <= _snap_num(s) <= n_zlast]
    snaps = snaps[::-1]  # newest first

    # --- Load z_last ONCE ---
    try:
        halos0 = halo_list(output_zlast, quiet=True)
    except Exception as e:
        print('[build_merger_tree] Could not load halo list at {0}: {1}'.format(output_zlast, e))
        return {hid: ([], []) for hid in halo_ids}

    params0 = _read_info_params(output_zlast)
    aexp0   = params0.get('aexp', 1.0)
    z0      = 1.0 / aexp0 - 1.0

    try:
        sim0 = _load_sim(output_zlast)
    except Exception as e:
        print('[build_merger_tree] Could not load particles: {0}'.format(e))
        return {hid: ([], []) for hid in halo_ids}

    pos_kpc0   = np.array(sim0['pos'])   # kpc
    iord_all0  = sim0['iord']
    tree0      = KDTree(pos_kpc0)

    # --- Init per-halo structures ---
    all_nodes   = {}
    all_edges   = {}
    watch_lists = {}

    for hid in halo_ids:
        row0 = halos0[halos0[:, 0] == hid]
        if len(row0) == 0:
            print('[build_merger_tree] halo_id {0} not found at {1}'.format(hid, output_zlast))
            continue
        row0       = row0[0]
        r200_0_kpc = _r200_kpc(row0[10], params0)
        hits0      = tree0.query_radius([row0[4:7]], r_search_factor * r200_0_kpc)[0]
        iord0      = iord_all0[hits0]
        if len(iord0) == 0:
            print('[build_merger_tree] No particles found in root halo {0}'.format(hid))
            continue
        root_node = {
            'snap':    output_zlast,
            'halo_id': int(row0[0]),
            'mass':    row0[10],
            'pos':     row0[4:7],   # kpc
            'z':       z0,
            'r200':    r200_0_kpc,  # kpc
            'is_main': True,
            'iord':    iord0,
        }
        all_nodes[hid]   = [root_node]
        all_edges[hid]   = []
        watch_lists[hid] = [(0, iord0)]
        print('[build_merger_tree] root: halo_id={0}  npart={1}  r200={2:.1f} kpc'.format(
            hid, len(iord0), r200_0_kpc))

    if not watch_lists:
        return {hid: ([], []) for hid in halo_ids}

    # --- Single backward pass through snapshots ---
    for snap in snaps[1:]:
        active = [hid for hid in watch_lists if watch_lists[hid]]
        if not active:
            break

        params       = _read_info_params(snap)
        aexp         = params.get('aexp', 1.0)
        z            = 1.0 / aexp - 1.0
        if z > z_max:
            print('[build_merger_tree] z={0:.2f} > z_max={1:.1f}, stopping'.format(z, z_max))
            break
        npart_thresh = _read_npart_threshold(snap)

        try:
            halos = halo_list(snap, quiet=True)
        except Exception:
            continue

        try:
            sim = _load_sim(snap)
        except Exception:
            continue

        pos_kpc   = np.array(sim['pos'])   # kpc
        iord_snap = sim['iord']

        for hid in active:
            new_watch = []
            for (desc_idx, tracked_iord) in watch_lists[hid]:

                # Step 1: find tracked particles in this snapshot by iord
                found_mask = np.isin(iord_snap, tracked_iord)
                n_found    = int(np.sum(found_mask))
                print('[build_merger_tree] halo={0}  snap={1}  z={2:.3f}  tracked={3}  found={4}'.format(
                    hid, os.path.basename(snap), z, len(tracked_iord), n_found))
                if not np.any(found_mask):
                    continue
                found_pos_kpc = pos_kpc[found_mask]   # kpc
                found_iord    = iord_snap[found_mask]

                # Step 2: for each halo collect indices of tracked particles within R200
                # halos cols: 0=idx, 2=parent, 4:7=pos_kpc, 10=mass_msol
                halo_parts = {}
                for hi, hrow in enumerate(halos):
                    r200_kpc = _r200_kpc(hrow[10], params)
                    dists    = np.linalg.norm(found_pos_kpc - hrow[4:7], axis=1)
                    inside   = np.where(dists <= r_search_factor * r200_kpc)[0]
                    if len(inside) > 0:
                        halo_parts[hi] = set(inside.tolist())

                if not halo_parts:
                    continue

                # Step 3: resolve overlaps — each particle assigned to exactly one halo
                part_halos = {}
                for hi, parts in halo_parts.items():
                    for p in parts:
                        part_halos.setdefault(p, []).append(hi)

                assigned = {hi: set() for hi in halo_parts}
                for p, his in part_halos.items():
                    if len(his) == 1:
                        assigned[his[0]].add(p)
                        continue
                    # Resolve host/subhalo: discard host if a subhalo also claims the particle
                    his_set = set(his)
                    changed = True
                    while changed:
                        changed = False
                        for hi in list(his_set):
                            parent_id = int(halos[hi, 2])
                            for hj in list(his_set):
                                if hi != hj and int(halos[hj, 0]) == parent_id:
                                    his_set.discard(hj)
                                    changed = True
                                    break
                            if changed:
                                break
                    # Peers: assign to nearest centre
                    if len(his_set) > 1:
                        best = min(his_set,
                                   key=lambda hi: np.linalg.norm(found_pos_kpc[p] - halos[hi, 4:7]))
                        his_set = {best}
                    assigned[list(his_set)[0]].add(p)

                # Step 4: drop halos below threshold, sort by particle count descending
                valid = [(hi, parts) for hi, parts in assigned.items()
                         if len(parts) >= npart_thresh]
                print('[build_merger_tree]   halos claiming={0}  above threshold({1})={2}'.format(
                    len(assigned), npart_thresh, len(valid)))
                if not valid:
                    continue
                valid.sort(key=lambda x: len(x[1]), reverse=True)
                for rank, (hi, parts) in enumerate(valid):
                    print('[build_merger_tree]   {0} halo_id={1}  npart={2}'.format(
                        'main' if rank == 0 else 'merger', int(halos[hi, 0]), len(parts)))

                # Step 5: record nodes and edges
                nodes = all_nodes[hid]
                edges = all_edges[hid]
                for rank, (hi, parts) in enumerate(valid):
                    hrow      = halos[hi]
                    part_idx  = np.array(list(parts))
                    r200_kpc  = _r200_kpc(hrow[10], params)
                    com_kpc   = np.mean(found_pos_kpc[part_idx], axis=0)
                    is_main   = (rank == 0)
                    node_idx  = len(nodes)
                    nodes.append({
                        'snap':    snap,
                        'halo_id': int(hrow[0]),
                        'mass':    hrow[10],
                        'pos':     com_kpc,    # kpc
                        'z':       z,
                        'r200':    r200_kpc,   # kpc
                        'is_main': is_main,
                        'iord':    found_iord[part_idx],
                    })
                    edges.append((node_idx, desc_idx, 'main' if is_main else 'merger'))
                    new_watch.append((node_idx, found_iord[part_idx]))

            watch_lists[hid] = new_watch

    return {hid: (all_nodes.get(hid, []), all_edges.get(hid, [])) for hid in halo_ids}


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
        m200_all = np.array([float(np.sum(sim_zlast['mass'][virial_zlast[i]].in_units('Msol'))) for i in range(wh1[0].size)])
        m200_mask = m200_all >= p.min_mass
        color_idx_map = np.cumsum(m200_mask) - 1
        halo_colors = sns.color_palette("husl",int(m200_mask.sum()))
        hull_vols = []
        hull_dens_vals = []
        hull_halo_idx = []
        hull_halo_ids = []
        hull_m_ratio = []
        hull_n_neighbors = []
        hull_safety = []
        if(p.plot or p.full_analysis):
            cp = halo_colors
            ax=plot_candidates(d[candidates[0][wh1][m200_mask],:],sim_zlast,comoving=True)
            if(p.plot and not p.plot_traceback):
                pyplot.savefig(p.fname+'.pdf',dpi=100)
            print('| ------------------------------------------------------------')
        print('------------------------------------------------------------')
        for i in range(wh1[0].size):
            sys.stdout.flush()
            if not m200_mask[i]:
                print('| {0:3d} | m200 {1:.2e} Msol < min_mass; skipping'.format(i+1,m200_all[i]))
                print('| ------------------------------------------------------------')
                continue
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
            extents_kpc = [
                np.max(sim_zinit['x'][region_zinit])-np.min(sim_zinit['x'][region_zinit]),
                np.max(sim_zinit['y'][region_zinit])-np.min(sim_zinit['y'][region_zinit]),
                np.max(sim_zinit['z'][region_zinit])-np.min(sim_zinit['z'][region_zinit]),
            ]
            if any(e/box_kpc > 0.5 for e in extents_kpc):
                safety = True
            if(safety):
                print('|     | --- Traceback region lies in boundaries')
                print('| ------------------------------------------------------------')
                continue
            # MUSIC risk: check whether the padded Lagrangian region would exceed half
            # the box at any refinement level. At level L, MUSIC adds `padding` cells on
            # each side of the region; the padded fraction of the box is:
            #   raw_fraction + 2*padding / 2^L
            # If this exceeds 0.5 at any level the MUSIC "subgrid larger than half box"
            # error will be triggered. The coarsest levels (low L) are the most restrictive
            # because the padding cells represent a larger fraction of the box there.
            music_risk = False
            threshold = 0.5 - p.music_margin
            for L in range(p.levelmin+1, p.levelmax+1):
                cell_frac = 2.0**(-L)
                if any(e/box_kpc + 2*p.padding*cell_frac > threshold for e in extents_kpc):
                    music_risk = True
                    break
            if music_risk:
                print('|     | --- WARNING: padded Lagrangian region may exceed half the box in MUSIC (levelmin={0}, levelmax={1}, padding={2}, margin={3})'.format(p.levelmin,p.levelmax,p.padding,p.music_margin))
            if(r200[i]>0.):
                npart_r200 = len(virial_zlast[i])
            else:
                npart_r200 = 0
            m200 = m200_all[i]
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
                hull_halo_idx.append(color_idx_map[i])
                hull_halo_ids.append(int(d[candidates[0][wh1[0][i]], 0]))
                hull_m_ratio.append(mass_region/m200)
                neighb_masses = d[neighbors[wh1[0][i]], 10]
                hull_n_neighbors.append(int(np.sum(neighb_masses > p.neighb_mass_frac * mass_candidate)))
                hull_safety.append(music_risk)
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
                    ax[k].plot(xvals[np.append(hull2d.vertices,hull2d.vertices[0])],yvals[np.append(hull2d.vertices,hull2d.vertices[0])],'k-',lw=2,color=cp[color_idx_map[i]])
                    left=np.argmin(xvals[hull2d.vertices])

            print('|     | --- Convex Hull                        -> vol={0:.3e} dens={1:.3e}'.format(hull.volume,float(np.sum(sim_zinit['mass'][region_zinit])/hull.volume)))
            try:
                _hull_pts = sim_zinit['pos'][region_zinit][hull.vertices]/box_kpc
                # Add a tiny reproducible jitter (1e-7 in unit-box coords ~ 2.5 pc for a 25 Mpc box).
                # MUSIC uses a simple gift-wrapping convex hull algorithm that explicitly does not
                # handle degeneracies (nearly co-planar vertices). When hull.vertices are passed
                # in scipy's arbitrary index order, the gift-wrapping initialization can pick a
                # degenerate starting edge, producing outward-pointing face normals that cause
                # check_point() to return False for all grid cells -> all-zero ic_refmap -> no
                # AMR refinement in RAMSES. The perturbation places vertices in general position,
                # guaranteeing correct face orientation. Fixed seed ensures reproducibility.
                _rng = np.random.RandomState(seed=12345)
                _hull_pts = _hull_pts + _rng.randn(*_hull_pts.shape) * 1e-7
                np.savetxt((p.fname+'_'+str(color_idx_map[i]+1)).strip(), _hull_pts)
                print('|     | --- Particle list outputed to '+(p.fname+'_'+str(color_idx_map[i]+1)).strip())
            except:
                print('[Error] Cannot write file '+(p.fname+'_'+str(color_idx_map[i]+1)).strip())
                sys.exit()
            print('| ------------------------------------------------------------')
            sys.stdout.flush()
        if((p.plot)and(p.plot_traceback)):
            pyplot.savefig(p.fname+'.pdf',dpi=100)
        if(p.full_analysis):
            pdf = PdfPages(p.fname+'_analysis.pdf')
            pdf.savefig(ax[0].get_figure(),dpi=100)
            if(len(hull_vols)>0):
                fig_scatter,(ax_s0,ax_s1) = pyplot.subplots(1,2,figsize=(18,8))
                for k in range(len(hull_vols)):
                    c  = halo_colors[hull_halo_idx[k]]
                    lbl = str(hull_halo_idx[k]+1)
                    ax_s0.scatter(hull_vols[k],hull_dens_vals[k],color=c,s=100,zorder=5)
                    if hull_safety[k]:
                        ax_s0.scatter(hull_vols[k],hull_dens_vals[k],s=300,facecolors='none',edgecolors='red',linewidths=2,zorder=6)
                    ax_s0.annotate(lbl,(hull_vols[k],hull_dens_vals[k]),
                                   textcoords='offset points',xytext=(6,4),color=c)
                    ax_s1.scatter(hull_n_neighbors[k],hull_m_ratio[k],color=c,s=100,zorder=5)
                    ax_s1.annotate(lbl,(hull_n_neighbors[k],hull_m_ratio[k]),
                                   textcoords='offset points',xytext=(6,4),color=c)
                ax_s0.set_xscale('log')
                ax_s0.set_yscale('log')
                ax_s0.set_xlabel('Lagrangian volume [kpc$^3$]')
                ax_s0.set_ylabel('Lagrangian density [M$_\\odot$ kpc$^{-3}$]')
                ax_s0.set_title('Lagrangian region')
                if any(hull_safety):
                    ax_s0.scatter([],[],s=150,facecolors='none',edgecolors='red',linewidths=2,label='> half box (MUSIC risk)')
                    ax_s0.legend(fontsize=9)
                ax_s1.set_xlabel('N neighbours with $m>{{{0:.0f}}}\%\\,m_{{\\mathrm{{cand}}}}$'.format(p.neighb_mass_frac*100))
                ax_s1.set_ylabel('$m_\\mathrm{region}\\,/\\,m_{200}$')
                ax_s1.set_title('Environment')
                sns.despine()
                pyplot.tight_layout()
                pdf.savefig(fig_scatter,dpi=100)
                pyplot.close(fig_scatter)
            # Merger trees for candidates without MUSIC risk (skipped if merger_tree=False)
            if not p.merger_tree:
                print('| Merger tree disabled (merger_tree=False)')
                pdf.close()
                print('| Full analysis saved to {0}_analysis.pdf'.format(p.fname))
                return
            sim_dir_mt = os.path.dirname(os.path.abspath(p.output_zlast))
            params_mt  = _read_info_params(p.output_zlast)
            # Collect uncached halo IDs and build all trees in a single snapshot pass
            uncached_ids = [hull_halo_ids[k] for k in range(len(hull_vols))
                            if not hull_safety[k]
                            and _load_merger_tree(p.fname, hull_halo_ids[k])[0] is None]
            if uncached_ids:
                trees = build_merger_tree(sim_dir_mt, uncached_ids,
                                          p.output_zlast, p.output_zinit)
                for hid, (nodes_mt, edges_mt) in trees.items():
                    if nodes_mt:
                        _save_merger_tree(p.fname, hid, nodes_mt, edges_mt)
            for k in range(len(hull_vols)):
                if hull_safety[k]:
                    continue
                halo_id_mt = hull_halo_ids[k]
                nodes_mt, edges_mt = _load_merger_tree(p.fname, halo_id_mt)
                if not nodes_mt:
                    continue
                fig_mt, (ax_t, ax_m) = pyplot.subplots(1, 2, figsize=(18, 8))
                plot_merger_tree(nodes_mt, edges_mt, ax_t, ax_m, params_mt,
                                 halo_label=str(hull_halo_idx[k]+1))
                pyplot.tight_layout()
                pdf.savefig(fig_mt, dpi=100)
                pyplot.close(fig_mt)
            pdf.close()
            print('| Full analysis saved to {0}_analysis.pdf'.format(p.fname))

    else:
        print('| No haloes matching the criteria')
        return

    return


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
            self.halo_coords = (np.array(re.split(',|;',''.join(halo_coords_str.split())))).astype(float)
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
            self.ps = (np.array(re.split(',|;',''.join(ps_str.split())))).astype(int)
        except:
            self.ps = np.array([0,0,0]).astype(int)
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


def __halo_list_tracking(output,conf):
    list = glob.glob(output+'/clump_?????.txt?????')
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

    region_all_zoom = np.array([]).astype(int)
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
            # Build tree for halos (positions in kpc)
            tree_halo = KDTree(np.squeeze((hl[:,4:7])),leaf_size=p.tree_nleaves)
            diff = k-j+1
            if(j<nfiles):
                # Save previous snapshot
                sim_prev = sim_curr
                tree_part_prev = tree_part_curr
                _box_kpc_prev = _box_kpc_curr
                # Find halos matching coordinate filter around previous halo
                # x[k],y[k],z[k] stored in code units; convert to kpc for tree query
                halo_candidates = tree_halo.query_radius([x[k]*_box_kpc_prev,y[k]*_box_kpc_prev,z[k]*_box_kpc_prev],p.rvir_search*r200_start)[0]
                # Load current snapshot
                sim_curr = _load_sim(list[j-1])
                sim_curr = sim_curr[np.argsort(sim_curr['iord'])]
                aexp_curr = float(sim_curr.properties['a'])
                _box_kpc_curr = float(sim_curr.properties['boxsize'].in_units('kpc'))
                to_msol = float(np.sum(sim_curr['mass'].in_units('Msol')))
                mass_cutoff = max(p.halo_cutoff/to_msol,p.halo_massfrac*mass_curr)
                # Filter low mass halos
                halo_candidates = halo_candidates[hl[halo_candidates,10]>mass_cutoff]
                if(len(halo_candidates)==0):
                    print('| No halos found')
                    print('| Tracking stopped at aexp={0}'.format(aexp_curr))
                    break
                # Gather particles in the previous selected halo (r200_start in kpc)
                halo_part_prev = tree_part_prev.query_radius([x[k]*_box_kpc_prev,y[k]*_box_kpc_prev,z[k]*_box_kpc_prev],p.rvir_track*r200_start)[0]
                # Build tree for particles
                print('|    | npart tree               = {0:9d} ------------------'.format(len(sim_curr)))
                tree_part_curr = KDTree(np.squeeze((sim_curr['pos'])),leaf_size=p.tree_nleaves)
                print('|    | Previous halo population = {0:7d} --------------------'.format(len(halo_candidates)))
                print('|    | Cutoff mass              = {0:4.2e} -------------------'.format(mass_cutoff*to_msol))

                ids_frac = np.zeros(len(halo_candidates))
                ii = 0
                for halo in halo_candidates:
                    # Compute R200 in kpc from clump mass (code fraction units)
                    r200_candidate = (hl[halo,10]*3./(200.*4.*math.pi))**(1.0/3.0) * _box_kpc_curr
                    # Gather particles of the halo to track
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
                # Computing Virial radius of the best candidate (in kpc)
                _pos_curr  = np.array(sim_curr['pos'])
                _mass_curr = np.array(sim_curr['mass'].in_units('Msol'))
                try:
                    r200_curr = _virial_radius(_pos_curr,_mass_curr,_box_kpc_curr,hl[id,4:7],p.rvir_search*r200_curr)
                except:
                    print('| [Warning] Virial radius computation did not converge')
                    r200_curr = (hl[id,10]*3./(200.*4.*math.pi))**(1.0/3.0) * _box_kpc_curr
                mass_curr = hl[id,10]
                print('|    |    -->  halo {0:7d} selected'.format(id))
                print('| ------------------------------------------------------------')

            # Final snapshot - starting point
            else:
                print('| Closest halo coordinates  = [{0:.5f},{1:.5f},{2:.5f}] kpc'.format(hl[id,4],hl[id,5],hl[id,6]))
                if((p.halo_coords[0]>0.) & (p.halo_coords[1]>0.) & (p.halo_coords[2]>0.)):
                    print('| Relative distance         = {0:.2e} kpc'.format(np.min(dist_halo)))
                # Loading first snapshot
                sim_curr = _load_sim(list[j-1])
                aexp_curr = float(sim_curr.properties['a'])
                _box_kpc_curr = float(sim_curr.properties['boxsize'].in_units('kpc'))
                # Computing virial radius (in kpc)
                _pos_curr  = np.array(sim_curr['pos'])
                _mass_curr = np.array(sim_curr['mass'].in_units('Msol'))
                r200_start = _virial_radius(_pos_curr,_mass_curr,_box_kpc_curr,hl[id,4:7],0.5*_box_kpc_curr)
                r200_curr = r200_start
                mass_curr = hl[id,10]
                id_start = id
                to_msol = float(np.sum(sim_curr['mass'].in_units('Msol')))
                print('| R200                      = {0:.4f} kpc'.format(r200_start))
                print('| M200                      = {0:.2e} Msol'.format(hl[id,10]*to_msol))
                print('| coords                    = {0} kpc'.format(hl[id,4:7]))
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

            # Code to physical units (kept for printing; positions already in kpc)
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
            # r200_curr is already in kpc; comoving = physical/aexp
            print('| R200                      = {0:.1f} kpc physical / {1:.1f} kpc comoving'.format(r200_curr,r200_curr/aexp_curr))
            print('| M200                      = {0:.2e} Msol'.format(m200))
            print('| M_clump                   = {0:.2e} Msol'.format(mass_candidate*to_msol))
            print('| position                  = [{0:.4f},{1:.4f},{2:.4f}] kpc'.format(hl[id,4],hl[id,5],hl[id,6]))
            print('| ------------------------------------------------------------')
            print('| npart_tot(r<R200)         = {1}'.format(p.rvir,len(virial_curr)))
            print('| npart_tot(r<{0}*R200)     = {1}'.format(p.rvir,len(region_curr)))
            print('| npart_coarse(r<{0}*R200)  = {1}'.format(p.rvir,len(coarse_in_rtb[0])))
            print('| npart_coarse_all          = {0}'.format(ncoarse_in_rtb_all))
            print('| contamination(r<{0}*R200) = {1:.1f}%'.format(p.rvir,100*float(np.sum(sim_curr['mass'][region_curr][coarse_in_rtb]))/float(np.sum(sim_curr['mass'][region_curr]))))
            print('| npart_zoom                = {0}'.format(len(zoom_part[0])))
            print('| npart_tot                 = {0}'.format(len(sim_curr)))
            # Get unique indices
            ind_curr = sim_zinit['iord'][region_all_zoom]
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
            # Compute centered positions and radii at z_init
            # (replaces pynbody in-place sim['pos'] centering + sim['r'])
            pos_zinit_cen = np.array(sim_zinit['pos']) - np.array(zinit_center)
            r_zinit = np.linalg.norm(pos_zinit_cen, axis=1)
            allowed = np.where((r_zinit[region_zinit]<p.rexclude)|(sim_zinit['mass'][region_zinit]<1.1*np.min(sim_zinit['mass'])))
            not_allowed = np.where((r_zinit[region_zinit]>=p.rexclude)&(sim_zinit['mass'][region_zinit]>1.1*np.min(sim_zinit['mass'])))
            print('| Included coarse part      = {0}'.format(len(allowed[0])))
            print('| Excluded coarse part      = {0}'.format(len(not_allowed[0])))
            if(len(coarse_in_rtb_init)>0):
                try:
                    # Compute centered positions and radii at curr snapshot
                    # (replaces pynbody in-place sim['pos'] centering + sim['r'])
                    pos_curr_cen = np.array(sim_curr['pos']) - hl[id,4:7]
                    r_curr = np.linalg.norm(pos_curr_cen, axis=1)
                    print('| r_min coarse part/R200    = {0:.3e}'.format(float(np.min(r_curr[region_curr][coarse_in_rtb]))/r200_curr))
                    print('| r_mean coarse part/R200   = {0:.3e}'.format(float(np.mean(r_curr[region_curr][coarse_in_rtb]))/r200_curr))
                except:
                    pass
                # Computing convex hulls volumes
                hull = ConvexHull(np.array(sim_zinit['pos'])[region_zinit][allowed])
                hull_zoom = ConvexHull(np.array(sim_zinit['pos'])[zoom_init])
                print('| Convex Hull coarse part -> vol={0:.3e} dens={1:.3e}'.format(hull.volume,float(np.sum(sim_zinit['mass'][region_zinit][allowed])/hull.volume)))
                print('| Convex Hull zoom part   -> vol={0:.3e} dens={1:.3e}'.format(hull_zoom.volume,float(np.sum(sim_zinit['mass'][zoom_init])/hull_zoom.volume)))
                print('| Volume increase         -> {0:.2f}%'.format(100*(hull.volume/hull_zoom.volume)-100.))

            if((np.max(ids_frac)>0.01)&(aexp_curr>p.aexp_min)):
                # Store position in code units (0..1) for RAMSES polynomial fit
                x[j-1] = hl[id,4] / _box_kpc_curr
                y[j-1] = hl[id,5] / _box_kpc_curr
                z[j-1] = hl[id,6] / _box_kpc_curr
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
            np.savetxt((p.fname).strip()+'_part',np.array(sim_zinit['pos'])[region_zinit][allowed][hull.vertices]-shift)
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

    # Write results (x,y,z in code units 0..1 matching pynbody version)
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
                xmin_coarse_in_rtb = float(np.min(sim_zinit[x][region_zinit][coarse_in_rtb_init]))
                ymin_coarse_in_rtb = float(np.min(sim_zinit[y][region_zinit][coarse_in_rtb_init]))
                xmax_coarse_in_rtb = float(np.max(sim_zinit[x][region_zinit][coarse_in_rtb_init]))
                ymax_coarse_in_rtb = float(np.max(sim_zinit[y][region_zinit][coarse_in_rtb_init]))
            except:
                xmin_coarse_in_rtb = 1.0
                ymin_coarse_in_rtb = 1.0
                xmax_coarse_in_rtb = 0.0
                ymax_coarse_in_rtb = 0.0

            xmin = min(xmin_coarse_in_rtb,float(np.min(sim_zlast[x][zoom_part])))-0.01
            ymin = min(ymin_coarse_in_rtb,float(np.min(sim_zlast[y][zoom_part])))-0.01
            xmax = max(xmax_coarse_in_rtb,float(np.max(sim_zlast[x][zoom_part])))+0.01
            ymax = max(ymax_coarse_in_rtb,float(np.max(sim_zlast[y][zoom_part])))+0.01
            pmin = min(xmin,ymin)
            pmax = max(xmax,ymax)
            ax[i].set_xlim([pmin,pmax])
            ax[i].set_ylim([pmin,pmax])
            ax[i].set_xlabel(x+' [kpc]')
            ax[i].set_ylabel(y+' [kpc]')
            im,xedges,yedges = np.histogram2d(sim_zlast[x][zoom_part],sim_zlast[y][zoom_part],
                weights=np.array(sim_zlast['mass'])[zoom_part],bins=1024,range=[[pmin,pmax],[pmin,pmax]])
            im = np.rot90(im)
            # Plotting 2D Convex Hull
            points_2d = np.squeeze([[np.array(sim_zinit[x])[region_zinit][allowed]],
                [np.array(sim_zinit[y])[region_zinit][allowed]]]).transpose()
            hull2d = ConvexHull(points_2d)
            ax[i].plot(np.array(sim_zinit[x])[region_zinit][allowed][np.append(hull2d.vertices,hull2d.vertices[0])],
                np.array(sim_zinit[y])[region_zinit][allowed][np.append(hull2d.vertices,hull2d.vertices[0])],
                'k-',lw=1.,color=cp[5],label='Lagrangian volume')
            points_2d = np.squeeze([[np.array(sim_zinit[x])[zoom_init]],[np.array(sim_zinit[y])[zoom_init]]]).transpose()
            hull2d = ConvexHull(points_2d)
            ax[i].plot(np.array(sim_zinit[x])[zoom_init][np.append(hull2d.vertices,hull2d.vertices[0])],
                np.array(sim_zinit[y])[zoom_init][np.append(hull2d.vertices,hull2d.vertices[0])],
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
                points_2d = np.vstack((np.array(sim_zinit[x])[region_zinit][coarse_in_rtb_init],np.array(sim_zinit[y])[region_zinit][coarse_in_rtb_init])).transpose()
                points_2d = np.round(points_2d*2000.)/2000.
                unique_points_2d = __unique_rows(points_2d)
                ax[i].scatter(unique_points_2d[:,0],unique_points_2d[:,1],
                    c=cp[4],marker='+',s=5,alpha=0.50,linewidth=0.5,label='Contaminating part initial')
            tv = ax[i].imshow(np.log10(im),cmap='bone_r',interpolation='quadric',aspect='equal',extent=[pmin,pmax,pmin,pmax])
        ax[0].legend(loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.10, 1.0, 0.1),ncol=2,markerscale=5.)
        out=p.fname+'_decontamination.pdf'
        pyplot.savefig(out,dpi=100)
