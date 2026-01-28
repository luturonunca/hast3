import glob
import math
import sys
import warnings
import configparser

import numpy as np
import yt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore")

mpl_major = int(matplotlib.__version__[0])
mpl_minor = int(matplotlib.__version__[2])
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cp = sns.color_palette(flatui)

if (mpl_major >= 2 or (mpl_major == 1 and mpl_minor >= 5)):
    mpl_colormap = "plasma"
else:
    mpl_colormap = "gist_heat"


def __version():
    print("     ___           ___           ___                  ")
    print("    /\\  \\         /\\  \\         /\\__\\                 ")
    print("    \\:\\  \\       /::\\  \\       /:/ _/_         ___    ")
    print("     \\:\\  \\     /:/\\:\\  \\     /:/ /\\  \\       /\\__\\   ")
    print(" ___ /::\\  \\   /:/ /::\\  \\   /:/ /::\\  \\     /:/  /   ")
    print("/\\  /:/\\:\\__\\ /:/_/:/\\:\\__\\ /:/_/:/\\:\\__\\   /:/__/    ")
    print("\\:\\/::/  \/__/ \\:\\/::/  \/__/ \\:\\/::/ /:/  /  /::\\  \\    ")
    print(" \\::/__/       \\::/__/       \\::/ /:/  /  /:/\\:\\  \\   ")
    print("  \\:\\  \\        \\:\\  \\        \\/_/:/  /   \/__\\:\\  \\  ")
    print("   \\:\\__\\        \\:\\__\\         /:/  /         \\:\\__\\ ")
    print("    \\/__/         \\/__/         \\/__/           \\/__/ ")
    print("| ------------------------------------------------------------")
    print("| HAlo Selection Tools - Version 0.5")


class config_selection_obj():
    def parse_input(self, ConfigFile):
        config = configparser.SafeConfigParser({"fname": "music_zoom", "recompute_rtb": False, "plot": False})
        config.read(ConfigFile)

        self.output_zinit = config.get("selection", "output_zinit")
        self.output_zlast = config.get("selection", "output_zlast")
        self.min_mass = config.getfloat("selection", "min_mass")
        self.max_mass = config.getfloat("selection", "max_mass")
        self.max_mass_neighb = config.getfloat("selection", "max_mass_neighb")
        self.rtb = config.getfloat("selection", "rtb")
        self.rbuffer = config.getfloat("selection", "rbuffer")
        try:
            self.xsearch = config.getfloat("selection", "xsearch")
            self.ysearch = config.getfloat("selection", "ysearch")
            self.zsearch = config.getfloat("selection", "zsearch")
            self.rsearch = config.getfloat("selection", "rsearch")
        except:
            self.xsearch = 0.5
            self.ysearch = 0.5
            self.zsearch = 0.5
            self.rsearch = -1.0
        try:
            self.min_neighb = config.getint("selection", "min_neighb")
        except:
            self.min_neighb = 0
        try:
            self.max_neighb = config.getint("selection", "max_neighb")
        except:
            self.max_neighb = 100000
        self.fname = config.get("selection", "fname")
        try:
            self.plot = config.getboolean("selection", "plot")
        except:
            self.plot = True
        try:
            self.plot_traceback = config.getboolean("selection", "plot_traceback")
        except:
            self.plot_traceback = False
        try:
            self.tree_nleaves = config.getint("selection", "tree_nleaves")
        except:
            self.tree_nleaves = 100
        try:
            self.boundary_min = config.getfloat("selection", "boundary_min")
        except:
            self.boundary_min = 0.1
        try:
            self.boundary_max = config.getfloat("selection", "boundary_max")
        except:
            self.boundary_max = 0.9
        try:
            self.clump_mass_unit = config.get("selection", "clump_mass_unit")
        except:
            self.clump_mass_unit = "fraction"
        try:
            self.create_halo_catalog = config.getboolean("selection", "create_halo_catalog")
        except:
            self.create_halo_catalog = False
        try:
            self.halo_finder = config.get("selection", "halo_finder")
        except:
            self.halo_finder = "hop"
        try:
            self.min_halo_particles = config.getint("selection", "min_halo_particles")
        except:
            self.min_halo_particles = 0
        try:
            self.hop_threshold = config.getfloat("selection", "hop_threshold")
        except:
            self.hop_threshold = 0.0


def _find_ramses_info(path):
    if path.endswith(".txt") and "info_" in path:
        return path
    info = glob.glob(path.rstrip("/") + "/info_*.txt")
    if len(info) > 0:
        return sorted(info)[-1]
    return None


def _load_ds(path):
    info_path = _find_ramses_info(path)
    if info_path is None:
        raise IOError("No info_*.txt found in {0}".format(path))
    ds = yt.load(info_path)
    return ds


def _get_particle_data(ds):
    ad = ds.all_data()
    pos_x = ad[("all", "particle_position_x")].to("kpc")
    pos_y = ad[("all", "particle_position_y")].to("kpc")
    pos_z = ad[("all", "particle_position_z")].to("kpc")
    mass = ad[("all", "particle_mass")].to("Msun")
    iord = np.array(ad[("all", "particle_identity")]).astype(np.int64)

    pos = np.vstack((pos_x.to_value("kpc"), pos_y.to_value("kpc"), pos_z.to_value("kpc"))).T
    mass_msol = mass.to_value("Msun")

    try:
        boxsize_kpc = float(ds.domain_width[0].to("kpc"))
    except Exception:
        boxsize_kpc = float(ds.domain_width[0])

    try:
        aexp = float(ds.scale_factor)
    except Exception:
        aexp = 1.0

    return {
        "pos": pos,
        "mass": mass_msol,
        "iord": iord,
        "boxsize_kpc": boxsize_kpc,
        "aexp": aexp,
    }


def _clump_header_format(path):
    try:
        with open(path, "r") as f:
            header = f.readline().strip()
        header = header.lstrip("#").strip().lower()
        if "peak_x" in header and "mass_cl" in header:
            return "new"
    except Exception:
        pass
    return "legacy"


def _normalize_clump_columns(data, fmt):
    if fmt != "new":
        return data
    if data.shape[1] < 12:
        return data
    data = data.copy()
    data[:, 3] = data[:, 4]
    data[:, 4:7] = data[:, 5:8]
    data[:, 8] = data[:, 9]
    data[:, 9] = data[:, 10]
    data[:, 10] = data[:, 11]
    return data


def halo_list(output, sim, quiet=False, clump_mass_unit="fraction"):
    files = glob.glob(output + "/clump_?????.txt?????")
    if len(files) == 0:
        print("[Error] No clump_*.txt* files found in {0}".format(output))
        sys.exit()

    if not quiet:
        print("| ------------------------------------------------------------")
        print("| Reading RAMSES clump finder files")
        print("| ------------------------------------------------------------")
        print("| nfiles        = {0}".format(len(files)))

    fmt = _clump_header_format(files[0])
    i = 0
    for file in files:
        data = np.loadtxt(file, skiprows=1, dtype=None)
        if np.size(data) == 0:
            continue
        if i > 0:
            data_all = np.vstack((data_all, data))
        else:
            data_all = data
        i = i + 1

    data_all = _normalize_clump_columns(data_all, fmt)
    data_sorted = data_all[data_all[:, 10].argsort()]

    if sim["boxsize_kpc"] is not None:
        if np.max(data_sorted[:, 4:7]) <= 1.0:
            data_sorted[:, 4:7] *= sim["boxsize_kpc"]

    total_mass = float(np.sum(sim["mass"]))
    particle_mass = float(np.min(sim["mass"]))
    if clump_mass_unit == "fraction":
        data_sorted[:, 10] *= total_mass
    elif clump_mass_unit in ("particle", "particles"):
        data_sorted[:, 10] *= particle_mass
    elif clump_mass_unit == "msol":
        pass
    else:
        if not quiet:
            print("[Warning] Unknown clump_mass_unit={0}; using fraction".format(clump_mass_unit))
        data_sorted[:, 10] *= total_mass

    if not quiet:
        min_m = np.min(data_sorted[:, 10])
        max_m = np.max(data_sorted[:, 10])
        min_part_mass = float(np.min(sim["mass"]))
        max_part_mass = float(np.max(sim["mass"]))
        print("| Min mass      = {0:.2e} Msol".format(min_m))
        print("| Max mass      = {0:.2e} Msol".format(max_m))
        print("| Min part mass = {0:.3e} Msol".format(min_part_mass))
        print("| Max part mass = {0:.3e} Msol".format(max_part_mass))
        print("| Total mass    = {0:.2e} Msol".format(total_mass))
        print("| ------------------------------------------------------------")

    return data_sorted


def plot_candidates(data, sim, center=[0.0, 0.0, 0.0], comoving=False, show_points=True):
    sns.set_context("poster")
    sns.set_style("ticks", {"axes.grid": False, "xtick.direction": "in", "ytick.direction": "in"})
    cp2 = sns.color_palette("Set1", len(data[:, 0]))
    print("| Plotting ", len(data[:, 0]), " haloes")
    fig, ax = pyplot.subplots(1, 2, figsize=(18, 8), sharex=True)
    proj = [["y", "x"], ["z", "x"]]
    dproj = [[5, 4], [6, 4]]

    plot_scale = 1.0
    if comoving:
        aexp = sim["aexp"]
        data_plot = data.copy()
        data_plot[:, 4:7] /= aexp
        sim_x = sim["x"] / aexp
        sim_y = sim["y"] / aexp
        sim_z = sim["z"] / aexp
    else:
        data_plot = data
        sim_x = sim["x"]
        sim_y = sim["y"]
        sim_z = sim["z"]

    for i in range(len(ax)):
        x = proj[i][0]
        y = proj[i][1]
        if np.max(sim_x) > 1.0 or np.max(sim_y) > 1.0:
            if comoving:
                unit_label = " [Mpc comov]"
            else:
                unit_label = " [Mpc]"
            plot_scale = 1000.0
        else:
            unit_label = " [code]"
        ax[i].set_xlabel(x + unit_label)
        ax[i].set_ylabel(y + unit_label)
        if np.max(sim_x) > 1.0 or np.max(sim_y) > 1.0:
            boxsize = sim["boxsize_kpc"]
            if comoving:
                boxsize /= sim["aexp"]
            boxsize /= plot_scale
            hist_range = [[0.0, boxsize], [0.0, boxsize]]
        else:
            hist_range = [[0.0, 1.0], [0.0, 1.0]]

        if x == "x":
            sim_x_plot = sim_x
        elif x == "y":
            sim_x_plot = sim_y
        else:
            sim_x_plot = sim_z

        if y == "x":
            sim_y_plot = sim_x
        elif y == "y":
            sim_y_plot = sim_y
        else:
            sim_y_plot = sim_z

        im, xedges, yedges = np.histogram2d(
            sim_x_plot / plot_scale,
            sim_y_plot / plot_scale,
            weights=sim["mass"],
            bins=512,
            range=hist_range,
        )
        im = np.rot90(im)
        data_plot[:, 4:7] -= center
        data_plot[:, 4:7] /= plot_scale
        if show_points:
            ax[i].scatter(
                data_plot[:, dproj[i][0]],
                data_plot[:, dproj[i][1]],
                s=50,
                c=cp2,
                alpha=0.5,
            )
        ax[i].set(adjustable="box", aspect="equal")
        extent_max = hist_range[0][1]
        ax[i].imshow(
            np.log10(im),
            cmap="bone_r",
            interpolation="quadric",
            aspect="equal",
            extent=[0.0, extent_max, 0.0, extent_max],
        )
        ax[i].set_xlim([0.0 - center[0] / plot_scale, extent_max - center[0] / plot_scale])
        ax[i].set_ylim([0.0 - center[1] / plot_scale, extent_max - center[1] / plot_scale])
        if show_points:
            for j in range(len(data_plot[:, 0])):
                ax[i].annotate(
                    str(j + 1),
                    (data_plot[j, dproj[i][0]] + 0.01, data_plot[j, dproj[i][1]] + 0.01),
                    color=cp2[j],
                )

    pyplot.tight_layout()
    return ax


def find_region(data, radius, nregion):
    x = np.squeeze(data[:, 4:7])
    print("| Building Tree with {0} haloes".format(len(data[:, 0])))
    tree = KDTree(x)
    np.random.seed(0)
    print("| Querying halo Tree")
    rp = np.random.random((nregion, 3))
    res = tree.query_radius(rp, radius)
    return rp, res


def find_galaxy(data, radius, min_mass, max_mass):
    x = np.squeeze(data[:, 4:7])
    print("| Building Tree with {0} haloes".format(len(data[:, 0])))
    tree = KDTree(x)
    print("| Querying halo Tree")
    ok = np.where((data[:, 10] > min_mass) & (data[:, 10] < max_mass))
    if ok[0].size > 0:
        rp = np.squeeze(data[ok, 4:7])
        res = tree.query_radius(rp, radius)
    else:
        res = []
    del tree
    return ok, res


def _compute_r200(tree, pos, mass, center, r_max, mean_density, overdensity=200.0):
    idx = tree.query_radius(center.reshape(1, -1), r_max)[0]
    if idx.size == 0:
        return 0.0
    rel = pos[idx] - center
    r = np.linalg.norm(rel, axis=1)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = mass[idx][order]
    cum_mass = np.cumsum(m_sorted)

    valid = r_sorted > 0.0
    r_sorted = r_sorted[valid]
    cum_mass = cum_mass[valid]
    if r_sorted.size == 0:
        return 0.0

    density = cum_mass / ((4.0 / 3.0) * math.pi * r_sorted ** 3)
    target = overdensity * mean_density
    ok = np.where(density >= target)[0]
    if ok.size == 0:
        return 0.0
    return float(r_sorted[ok[-1]])


def select(config_file):
    __version()
    p = config_selection_obj()
    print("| ------------------------------------------------------------")
    print("| HAST - select_candidate")
    print("| ------------------------------------------------------------")
    try:
        p.parse_input(config_file)
    except:
        print("[Error] {0} file specified cannot be read".format(config_file))
        sys.exit()

    try:
        ds_zinit = _load_ds(p.output_zinit)
    except IOError:
        print("[Error] {0} file specified cannot be read".format(p.output_zinit))
        sys.exit()

    try:
        ds_zlast = _load_ds(p.output_zlast)
    except IOError:
        print("[Error] {0} file specified cannot be read".format(p.output_zlast))
        sys.exit()

    if p.create_halo_catalog:
        try:
            from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
        except Exception:
            print("[Error] yt halo_analysis not available; install yt_astro_analysis")
            sys.exit()
        print("| ------------------------------------------------------------")
        print("| Running {0} halo finder (yt)".format(p.halo_finder))
        print("| ------------------------------------------------------------")
        finder_kwargs = {}
        used_finder_min = False
        if p.halo_finder == "hop" and p.hop_threshold > 0.0:
            finder_kwargs["threshold"] = p.hop_threshold
        try:
            if finder_kwargs:
                hc = HaloCatalog(data_ds=ds_zlast, finder_method=p.halo_finder, finder_kwargs=finder_kwargs)
            else:
                hc = HaloCatalog(data_ds=ds_zlast, finder_method=p.halo_finder)
            hc.create()
            used_finder_min = bool(finder_kwargs)
        except TypeError as e:
            if finder_kwargs and "unexpected keyword argument" in str(e):
                print("[Warning] Halo finder does not accept finder_kwargs; proceeding without them")
                hc = HaloCatalog(data_ds=ds_zlast, finder_method=p.halo_finder)
                hc.create()
            else:
                raise
        if p.min_halo_particles > 0 and not used_finder_min:
            hc.add_filter("quantity_value", "particle_identifier", ">=", p.min_halo_particles)
            hc.create()

    if p.min_mass >= p.max_mass:
        print("[Error] min_mass>max_mass")
        sys.exit()

    sim_zinit = _get_particle_data(ds_zinit)
    sim_zlast = _get_particle_data(ds_zlast)

    for sim in (sim_zinit, sim_zlast):
        order = np.argsort(sim["iord"])
        sim["iord"] = sim["iord"][order]
        sim["pos"] = sim["pos"][order]
        sim["mass"] = sim["mass"][order]

    sim_zinit["x"] = sim_zinit["pos"][:, 0]
    sim_zinit["y"] = sim_zinit["pos"][:, 1]
    sim_zinit["z"] = sim_zinit["pos"][:, 2]
    sim_zlast["x"] = sim_zlast["pos"][:, 0]
    sim_zlast["y"] = sim_zlast["pos"][:, 1]
    sim_zlast["z"] = sim_zlast["pos"][:, 2]

    z_init = abs(1.0 / sim_zinit["aexp"] - 1.0)
    z_last = abs(1.0 / sim_zlast["aexp"] - 1.0)
    to_kpc = sim_zlast["boxsize_kpc"]
    to_kpc_comov = sim_zlast["boxsize_kpc"] / sim_zlast["aexp"]

    print("| ------------------------------------------------------------")
    print("| Selection output = {0} [z={1:5.2f}]".format(p.output_zlast, z_last))
    print("| Initial output   = {0} [z={1:5.2f}]".format(p.output_zinit, z_init))
    print("| r_tb             = {0:.2f} R200 ".format(p.rtb))
    print("| r_buffer         = {0:.2f} Mpc".format(p.rbuffer))
    print("| m_candidate      = {0:.3e} Msol < m < {1:.3e} Msol".format(p.min_mass, p.max_mass))
    print("| n_neighbors      = {0} < n < {1}".format(p.min_neighb, p.max_neighb))
    print("| m_neighbor_max   = m < {0:.1e}*m_candidate ".format(p.max_mass_neighb))
    print("| ------------------------------------------------------------")
    sys.stdout.flush()

    rbuffer_kpc = p.rbuffer * 1e3
    d = halo_list(p.output_zlast, sim_zlast, clump_mass_unit=p.clump_mass_unit)
    candidates, neighbors = find_galaxy(d, rbuffer_kpc, p.min_mass, p.max_mass)
    nc = candidates[0].size
    print("| ------------------------------------------------------------")
    print("| Found {0} candidates for {1:.2e}<m<{2:.2e}".format(nc, p.min_mass, p.max_mass))
    if nc == 0:
        return

    flag = np.zeros(nc)

    xsearch = p.xsearch
    ysearch = p.ysearch
    zsearch = p.zsearch
    rsearch = p.rsearch
    if p.rsearch > 0.0 and np.max(d[:, 4:7]) > 1.0:
        boxsize_kpc = sim_zlast["boxsize_kpc"]
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
    if np.max(d[:, 4:7]) > 1.0:
        boundary_min *= sim_zlast["boxsize_kpc"]
        boundary_max *= sim_zlast["boxsize_kpc"]

    for i in range(nc):
        if len(neighbors[i]) > p.max_neighb:
            flag[i] = 1
        if len(neighbors[i]) < p.min_neighb:
            flag[i] = 2
        nb = len(neighbors[i])
        for j in range(nb):
            if (d[neighbors[i][j], 10] > p.max_mass_neighb * d[candidates[0][i], 10]) and (
                neighbors[i][j] != candidates[0][i]
            ):
                flag[i] = 3
        if (d[candidates[0][i], 4] < boundary_min) or (d[candidates[0][i], 4] > boundary_max):
            # print(
            #     "| candidate {0} x = {1} outside [{2}, {3}]".format(
            #         i, d[candidates[0][i], 4], boundary_min, boundary_max
            #     )
            # )
            flag[i] = 4
        if (d[candidates[0][i], 5] < boundary_min) or (d[candidates[0][i], 5] > boundary_max):
            # print(
            #     "| candidate {0} y = {1} outside [{2}, {3}]".format(
            #         i, d[candidates[0][i], 5], boundary_min, boundary_max
            #     )
            # )
            flag[i] = 4
        if (d[candidates[0][i], 6] < boundary_min) or (d[candidates[0][i], 6] > boundary_max):
            # print("| candidate {0} z = {1} outside [{2}, {3}]".format(
            #     i, d[candidates[0][i], 6], boundary_min, boundary_max))
            flag[i] = 4
        if rsearch > 0.0:
            rfilter = math.sqrt(
                (d[candidates[0][i], 4] - xsearch) ** 2
                + (d[candidates[0][i], 5] - ysearch) ** 2
                + (d[candidates[0][i], 6] - zsearch) ** 2
            )
            if rfilter > rsearch:
                flag[i] = 5

    wh1 = np.where(flag == 0)
    wh2 = np.where(flag == 1)
    wh3 = np.where(flag == 2)
    wh4 = np.where(flag == 3)
    wh5 = np.where(flag == 4)
    if p.rsearch > 0.0:
        wh6 = np.where(flag == 5)
    print("| ------------------------------------------------------------")
    print("| {0:5d} valid candidates".format(wh1[0].size))
    print("| {0:5d} candidates with n_neighbor>{1}".format(wh2[0].size, p.max_neighb))
    print("| {0:5d} candidates with n_neighbor<{1}".format(wh3[0].size, p.min_neighb))
    print("| {0:5d} candidates with m_neighbor>{1:.2f}*m_candidate".format(wh4[0].size, p.max_mass_neighb))
    print("| {0:5d} candidates close to the box boundaries".format(wh5[0].size))
    if p.rsearch > 0.0:
        print("| {0:5d} outside of the search region".format(wh6[0].size))
    print("| ------------------------------------------------------------")
    sys.stdout.flush()

    if wh1[0].size > 0:
        if p.plot:
            cp_trace = sns.color_palette("Set1", wh1[0].size)
            ax = plot_candidates(d[candidates[0][wh1], :], sim_zlast, comoving=True, show_points=False)
            finder_label = p.halo_finder if p.create_halo_catalog else "ramses clump"
            text = (
                "finder: {0}\n"
                "m_range: {1:.1e}–{2:.1e} Msun\n"
                "r_tb: {3:.2f} R200\n"
                "r_buffer: {4:.2f} Mpc".format(
                    finder_label, p.min_mass, p.max_mass, p.rtb, p.rbuffer
                )
            )
            ax[0].text(
                0.02,
                0.98,
                text,
                transform=ax[0].transAxes,
                va="top",
                ha="left",
                fontsize="small",
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
            print("| ------------------------------------------------------------")

        print("| Building Tree [{0} particles]".format(len(sim_zlast["pos"])))
        tree = KDTree(sim_zlast["pos"], leaf_size=p.tree_nleaves)

        print("| Computing Virial radii")
        mean_density = np.sum(sim_zlast["mass"]) / (sim_zlast["boxsize_kpc"] ** 3)
        r200 = np.array([])
        for i in range(wh1[0].size):
            try:
                rr = _compute_r200(
                    tree,
                    sim_zlast["pos"],
                    sim_zlast["mass"],
                    d[candidates[0][wh1[0][i]], 4:7],
                    rbuffer_kpc,
                    mean_density,
                )
            except Exception:
                print("| [Warning] Virial radius computation did not converge")
                rr = 0.0
            r200 = np.append(r200, rr)

        print("| Querying particle Tree")
        if p.plot:
            if np.max(sim_zlast["pos"]) > 1.0:
                offset = 0.01 * (sim_zlast["boxsize_kpc"] / sim_zlast["aexp"]) / 1000.0
            else:
                offset = 0.01
            for i in range(wh1[0].size):
                center = d[candidates[0][wh1[0][i]], 4:7] / sim_zlast["aexp"] / 1000.0
                radius = r200[i] / sim_zlast["aexp"] / 1000.0
                ax[0].add_patch(
                    Circle(
                        (center[1], center[0]),
                        radius,
                        fill=True,
                        facecolor=cp_trace[i],
                        edgecolor=cp_trace[i],
                        lw=1.5,
                        alpha=0.5,
                    )
                )
                ax[1].add_patch(
                    Circle(
                        (center[2], center[0]),
                        radius,
                        fill=True,
                        facecolor=cp_trace[i],
                        edgecolor=cp_trace[i],
                        lw=1.5,
                        alpha=0.5,
                    )
                )
                ax[0].annotate(
                    str(i + 1),
                    (center[1] + offset, center[0] + offset),
                    color=cp_trace[i],
                )
                ax[1].annotate(
                    str(i + 1),
                    (center[2] + offset, center[0] + offset),
                    color=cp_trace[i],
                )
            if not p.plot_traceback:
                pyplot.savefig(p.fname + ".pdf", dpi=100)
        region_zlast = tree.query_radius(d[candidates[0][wh1], 4:7], p.rtb * r200)
        virial_zlast = tree.query_radius(d[candidates[0][wh1], 4:7], r200)
        print("------------------------------------------------------------")
        for i in range(wh1[0].size):
            sys.stdout.flush()
            ind_zlast = sim_zlast["iord"][region_zlast[i]]
            mass_region = float(np.sum(sim_zlast["mass"][region_zlast[i]]))
            mass_neighb = np.sum(d[neighbors[wh1[0][i]], 10])
            mass_candidate = d[candidates[0][wh1[0][i]], 10]
            pos_candidate = np.squeeze(d[candidates[0][wh1[0][i]], 4:7])

            region_zinit = np.searchsorted(sim_zinit["iord"], ind_zlast, side="left")
            npart = len(region_zinit)
            print(
                "| {0:3d} | m_candidate={1:.2e} Msol | {2} neighbors | m_region={3:.2e} Msol | npart={4} ".format(
                    i + 1, mass_candidate, len(neighbors[wh1[0][i]]), mass_region, npart
                )
            )
            safety = False
            box_kpc = sim_zinit["boxsize_kpc"]
            if (
                (np.max(sim_zinit["x"][region_zinit]) - np.min(sim_zinit["x"][region_zinit]))
                / box_kpc
                > 0.5
            ):
                safety = True
            if (
                (np.max(sim_zinit["y"][region_zinit]) - np.min(sim_zinit["y"][region_zinit]))
                / box_kpc
                > 0.5
            ):
                safety = True
            if (
                (np.max(sim_zinit["z"][region_zinit]) - np.min(sim_zinit["z"][region_zinit]))
                / box_kpc
                > 0.5
            ):
                safety = True
            if safety:
                print("|     | --- Traceback region lies in boundaries")
                print("| ------------------------------------------------------------")
                continue

            if r200[i] > 0.0:
                npart_r200 = len(virial_zlast[i])
            else:
                npart_r200 = 0
            m200 = float(np.sum(sim_zlast["mass"][virial_zlast[i]]))
            lambda200 = 0.0

            xmean = float(np.mean(sim_zinit["x"][region_zinit]))
            ymean = float(np.mean(sim_zinit["y"][region_zinit]))
            zmean = float(np.mean(sim_zinit["z"][region_zinit]))
            print("|     | --- Candidate halo properties")
            print("|     | --------------- m200                   -> {0:.3e} Msol".format(m200))
            print(
                "|     | --------------- r200                   -> [{0:.1f} kpc phys,{1:.1f} kpc comov]".format(
                    r200[i], r200[i] / sim_zlast["aexp"]
                )
            )
            print("|     | --------------- lambda                 -> {0:.4f}".format(lambda200))
            print("|     | --------------- npart(r<r200)          -> {0}".format(npart_r200))
            print(
                "|     | --- Candidate halo position            -> [{0:.5f},{1:.5f},{2:.5f}]".format(
                    pos_candidate[0], pos_candidate[1], pos_candidate[2]
                )
            )
            print(
                "|     | --- Mean particle position in ICs      -> [{0:.5f},{1:.5f},{2:.5f}]".format(
                    xmean, ymean, zmean
                )
            )
            hull = ConvexHull(sim_zinit["pos"][region_zinit] - sim_zinit["pos"][region_zinit].mean(axis=0))

            if (p.plot) and (p.plot_traceback):
                proj = [["y", "x"], ["z", "x"]]
                dproj = [[5, 4], [6, 4]]
                for k in range(len(ax)):
                    x = proj[k][0]
                    y = proj[k][1]
                    points_2d = np.squeeze(
                        [[sim_zinit[x][region_zinit] / sim_zinit["aexp"]], [sim_zinit[y][region_zinit] / sim_zinit["aexp"]]]
                    ).transpose()
                    hull2d = ConvexHull(points_2d)
                    aexp_init = sim_zinit["aexp"]
                    xvals = sim_zinit[x][region_zinit] / aexp_init
                    yvals = sim_zinit[y][region_zinit] / aexp_init
                    ax[k].plot(
                        xvals[np.append(hull2d.vertices, hull2d.vertices[0])],
                        yvals[np.append(hull2d.vertices, hull2d.vertices[0])],
                        "k-",
                        lw=2,
                        color=cp_trace[i],
                    )
                    left = np.argmin(xvals[hull2d.vertices])

            print(
                "|     | --- Convex Hull                        -> vol={0:.3e} dens={1:.3e}".format(
                    hull.volume, float(np.sum(sim_zinit["mass"][region_zinit]) / hull.volume)
                )
            )
            try:
                np.savetxt((p.fname + "_" + str(i + 1)).strip(), sim_zinit["pos"][region_zinit][hull.vertices])
                print("|     | --- Particle list outputed to " + (p.fname + "_" + str(i + 1)).strip())
            except Exception:
                print("[Error] Cannot write file " + (p.fname + "_" + str(i + 1)).strip())
                sys.exit()
            print("| ------------------------------------------------------------")
            sys.stdout.flush()

        if (p.plot) and (p.plot_traceback):
            pyplot.savefig(p.fname + ".pdf", dpi=100)

    else:
        print("| No haloes matching the criteria")
        return

    return
