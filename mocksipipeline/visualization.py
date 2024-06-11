"""
Visualization convenience functions for MOXSI data
"""
import astropy.units as u
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.visualization import (AsymmetricPercentileInterval,
                                   ImageNormalize, LogStretch)
from astropy.wcs.utils import wcs_to_celestial_frame
from overlappy.util import color_lat_lon_axes
from sunpy.coordinates.utils import get_limb_coordinates

__all__ = [
    'annotate_lines',
    'plot_labeled_spectrum',
    'plot_detector_image',
    'add_arrow_from_coords',
]


def annotate_lines(axis, cube, source_location, line_list, component_wcs, cube_total=None, threshold=None, **kwargs):
    """
    Add labels for spectral line identifications on MOXSI spectra plots

    Parameters
    ----------
    axis
    cube
    source_location
    line_list
    component_wcs
    """
    draggable = kwargs.pop('draggable', False)
    ratio_threshold = kwargs.pop('ratio_threshold')
    color = kwargs.get('color', 'k')
    annotate_kwargs = {
        'xytext': (0, 150),
        'textcoords': 'offset points',
        'rotation': 90,
        'color': color,
        'horizontalalignment': 'center',
        'verticalalignment':'center',
        'arrowprops': dict(color=color, arrowstyle='-[', ls='--'),
    }
    annotate_kwargs.update(kwargs)

    line_pos, _, _ = component_wcs.world_to_pixel(source_location, line_list['wavelength'])
    _, _, line_index = component_wcs.world_to_array_index(source_location, line_list['wavelength'])
    line_index = np.array(line_index)

    # NOTE: we cannot label lines for which we do not have corresponding data values
    in_bounds = np.where(np.logical_and(line_index>=0, line_index<cube.data.shape[0]))
    line_pos = line_pos[in_bounds]
    line_index = line_index[in_bounds]
    line_list = line_list[in_bounds]

    # NOTE: we don't want to waste time labeling lines that are below a certain threshold
    if threshold is not None:
        above_thresh = np.where(u.Quantity(cube.data[line_index], cube.unit)>=threshold[line_index])
        line_pos = line_pos[above_thresh]
        line_index = line_index[above_thresh]
        line_list = line_list[above_thresh]
    if cube_total is not None and ratio_threshold is not None:
        ratio = cube.data / cube_total.data
        above_thresh = np.where(ratio[line_index]>=ratio_threshold)
        line_pos = line_pos[above_thresh]
        line_index = line_index[above_thresh]
        line_list = line_list[above_thresh]

    for pos, index, row in zip(line_pos, line_index, line_list):
        wave_label = row["wavelength"].to_string(format="latex_inline", precision=5)
        _annotate = axis.annotate(f'{row["ion name"]}, {wave_label}', (pos, cube.data[index]), **annotate_kwargs)
        _annotate.draggable(draggable)

    return axis


def plot_labeled_spectrum(spectra_components, line_list=None, source_location=None, labels=None, colors=None, **kwargs):
    """
    Plot a 1D spectra across the MOXSI detector, including components from multiple orders with selected spectral lines
    labeled

    Parameters
    ----------
    spectra_components: `ndcube.NDCollection`
        A list of `ndcube.NDCube` objects containing the spectra for each order component we want to plot.
        Each cube should have a 3D WCS but the data should only have the last dimension of length greater
        than 1. One of these components should have the key "all_components" that represents the sum of
        all components.
    line_list: `astropy.table.Table`, optional
        A list of all the lines to label. At a minimum, there must be a column labeled "wavelength" and "ion name"
    source_location: `astropy.coordinates.SkyCoord`, optional
        The location of the source of the emission.
    labels: `list`, optional
        Labels for each of the entries in ``spectra_components``
    kwargs: optional
        Other options for configuring axes limits and line annotations
    """
    x_lim = kwargs.pop('x_lim', None)
    y_lim = kwargs.pop('y_lim', None)
    log_y = kwargs.pop('log_y', True)
    threshold = kwargs.pop('threshold', None)
    figsize = kwargs.pop('figsize', (20,10))
    skip_component_labels = kwargs.pop('skip_component_labels', [])
    spectra_total = spectra_components['all_components']

    fig = kwargs.pop('figure', None)
    if fig is None:
        fig = plt.figure(figsize=figsize)  # Avoid creating figure if not needed
    ax = kwargs.pop('axes', None)
    for k,v in spectra_components.items():
        label = labels[k] if labels else k
        color = colors[k] if colors else None
        if ax is None:
            ax = v[0,0,:].plot(label=label, color=color)
        else:
            v[0,0,:].plot(axes=ax, label=label, color=color)
        if line_list and source_location and k not in skip_component_labels:
            annotate_lines(ax,
                           v[0,0,:],
                           source_location,
                           line_list,
                           v.wcs,
                           cube_total=spectra_total[0,0,:],
                           threshold=threshold,
                           color=ax.lines[-1].get_color(),
                           **kwargs)

    if log_y:
        ax.set_yscale('log')
    ax.set_xlim((1000, 2000) if x_lim is None else x_lim)
    if y_lim is None:
        y_lim = (0, spectra_total.data.max()*1.025)
    ax.set_ylim(y_lim)
    ax.set_ylabel(f"Flux [{spectra_total.unit:latex}]")
    ax.set_xlabel('Dispersion direction')
    ax.legend(loc=4 if log_y else 1, ncol=5)

    return fig, ax


def plot_detector_image(moxsi_collection,
                        prime_key='spectrogram_pinhole_0',
                        line_list=None,
                        draw_limb=True,
                        **kwargs):
    """
    Plot the full detector image with appropriate labels
    """
    # Parse inputs
    figsize = kwargs.get('figsize', (20,20))
    cmap = kwargs.get('cmap', 'hinodexrt')
    norm = kwargs.get('norm')
    prime_component = moxsi_collection[prime_key]
    # Build figure
    fig = plt.figure(figsize=figsize, layout='tight')
    ax = fig.add_subplot(projection=prime_component[0,...].wcs)
    if norm is None:
        _, vmax = AsymmetricPercentileInterval(1, 99.9).get_limits(prime_component.data[0,...])
        norm = ImageNormalize(vmin=0, vmax=vmax, stretch=LogStretch())
    prime_component[0,...].plot(axes=ax, cmap=cmap, interpolation='none', norm=norm)

    # Ticks and direction annotationes
    grid_color = kwargs.get('grid_color', 'w')
    color_lat_lon_axes(ax, lon_color=grid_color, lat_color=grid_color)
    ax.coords[0].set_ticklabel(rotation=90, color='k')
    ax.coords[1].set_ticklabel(color='k')
    ax.coords[0].grid(ls=':', color=grid_color)
    ax.coords[1].grid(ls=':', color=grid_color)
    ax.coords[1].set_axislabel('HPC Longitude', color='k')
    ax.coords[0].set_axislabel('HPC Latitude', color='k')
    for c in ax.coords:
        c.set_ticks(([-1000, 0, 1000]*u.arcsec).to('deg'))
        c.set_major_formatter('s')

    # Add directional arrow
    head_coord = (-1900, 400)*u.arcsec
    arrow_length = (0, 800)*u.arcsec
    add_arrow_from_coords(ax, head_coord-arrow_length, head_coord, color='C4', mutation_scale=25,)
    ax.text(
        *head_coord.to_value('deg'),
        'N',
        va='center',
        ha='left',
        color='C4',
        transform=ax.get_transform('world'),
    )

    # Add labels to filtergrams
    # FIXME: Pull this from the metadata once it exists there. This correspondence is not guaranteed to always
    # be true
    if kwargs.get('label_components', True):
        key_labels = [(f'filtergram_{i+1}_0', lab) for i, lab in enumerate(['Be-thin', 'Be-med', 'Be-thick', 'Al-poly'])]
        key_labels += [('spectrogram_pinhole_0', 'Spectrogram Pinhole'), ('spectrogram_slot_0', 'Spectrogram Slot')]
        for key, label in key_labels:
            fg = moxsi_collection[key]
            coord = SkyCoord(Tx=-1400*u.arcsec, Ty=0*u.arcsec, frame=wcs_to_celestial_frame(fg.wcs))
            pix_coord = fg[0].wcs.world_to_pixel(coord)
            ax.annotate(label, pix_coord, ha='center', va='bottom', color=kwargs.get('label_color','w'))

    if draw_limb:
        hpc_frame = wcs_to_celestial_frame(moxsi_collection['filtergram_1_0'].wcs)
        observer = hpc_frame.observer
        limb_coord = get_limb_coordinates(observer, resolution=500)
        zero_order_components = [
            'filtergram_1_0',
            'filtergram_2_0',
            'filtergram_3_0',
            'filtergram_4_0',
            'spectrogram_slot_0',
            'spectrogram_pinhole_0',
        ]
        limb_color = kwargs.get('limb_color', 'w')
        for k in zero_order_components:
            _wcs = moxsi_collection[k][0,...].wcs
            px, py = _wcs.world_to_pixel(limb_coord)
            ax.plot(px, py, ls='--', color=limb_color, lw=0.5)

    # Add wavelength annotations
    annotate_kw = {
        'textcoords': 'offset points',
        'color': 'w',
        'arrowprops': dict(color='w', arrowstyle='-|>', lw=1),
        'horizontalalignment':'center',
        'verticalalignment':'center',
        'rotation':90,
        'fontsize': plt.rcParams['xtick.labelsize']*0.8,
        'weight': 'bold',
    }
    annotate_kw.update(kwargs.get('annotate_kwargs', {}))

    if line_list:
        annot_pt = moxsi_collection['filtergram_1_0'][0,...].wcs.array_index_to_world(
            *np.unravel_index(moxsi_collection['filtergram_1_0'].data[0].argmax(),
                            moxsi_collection['filtergram_1_0'].data[0].shape)
        )
        ytext_nom = 70
        thresh = 0.05 * u.ph / (u.h * u.pix)
        for _wcs in [moxsi_collection['spectrogram_pinhole_-1'].wcs, moxsi_collection['spectrogram_pinhole_1'].wcs]:
            ytext = ytext_nom
            pos_previous = 0
            for group in line_list.group_by('MOXSI pixel').groups:
                i_sort = np.argsort(group[r'active\_region'])
                row = group[i_sort[-1]]
                if row[r'active\_region'] < thresh:
                    continue
                if np.fabs(row['MOXSI pixel'] - pos_previous) < 11:
                    ytext *= -1.2
                    #if annotate_kw['verticalalignment'] == 'bottom':
                else:
                    ytext = ytext_nom
                ax.annotate(
                    f'{row["ion name"]}',
                    xy=_wcs.world_to_pixel(annot_pt, row['wavelength'])[:2],
                    xytext=(0, ytext),
                    **annotate_kw
                )
                pos_previous = row['MOXSI pixel']

    cbar_kwargs = {
        'ax': ax,
        'orientation': 'horizontal',
        'location': 'top',
        'pad': 0.01,
        'aspect': 70,
        'extend': 'max' if norm.vmax<prime_component.data.max() else 'neither',
        'extendfrac': 0.02,
        'shrink': 0.8,
        'format': matplotlib.ticker.LogFormatterMathtext(base=10.0,),
        'ticks': [0, 100, 1000, 1e4],
        'label': f'[{prime_component.unit:latex_inline}]'
    }
    cbar_kwargs.update(kwargs.get('cbar_kwargs', {}))
    fig.colorbar(ax.get_images()[0], **cbar_kwargs)

    return fig


def add_arrow_from_coords(ax, tail, head, **arrow_kwargs):
    """
    Add an arrow to an axis, given the head and tail coordinates.
    """
    if tail.unit == u.pix:
        transform = 'pixel'
        end_unit = 'pixel'
    else:
        transform = 'world'
        end_unit = 'deg'
    arrow = matplotlib.patches.FancyArrowPatch(tail.to_value(end_unit),
                                               head.to_value(end_unit),
                                               transform=ax.get_transform(transform),
                                               **arrow_kwargs)
    ax.add_patch(arrow)
