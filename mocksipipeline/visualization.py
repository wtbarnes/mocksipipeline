"""
Visualization convenience functions for MOXSI data
"""
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['annotate_lines', 'plot_labeled_spectrum']


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
    color = kwargs.get('color', 'k')
    annotate_kwargs = {
        'xytext': (0, 150),
        'textcoords': 'offset points',
        'rotation': 90,
        'color': color,
        'horizontalalignment': 'center',
        'verticalalignment':'center',
        'arrowprops': dict(color=color, arrowstyle='-', ls='--'),
    }
    annotate_kwargs.update(kwargs)

    line_pos, _, _ = component_wcs.world_to_pixel(source_location, line_list['wavelength'])
    _, _, line_index = component_wcs.world_to_array_index(source_location, line_list['wavelength'])
    line_index = np.array(line_index)

    # NOTE: we cannot label lines for which we do not have corresponding data values
    in_bounds = np.where(line_index<cube.data.shape[0])
    line_pos = line_pos[in_bounds]
    line_index = line_index[in_bounds]
    line_list = line_list[in_bounds]

    # NOTE: we don't want to waste time labeling lines that are below a certain threshold
    if cube_total is not None and threshold is not None:
        ratio = cube.data / cube_total.data
        above_thresh = np.where(ratio[line_index]>threshold)
        line_pos = line_pos[above_thresh]
        line_index = line_index[above_thresh]
        line_list = line_list[above_thresh]

    for pos, index, row in zip(line_pos, line_index, line_list):
        wave_label = row["wavelength"].to_string(format="latex_inline", precision=5)
        axis.annotate(f'{row["ion name"]}, {wave_label}', (pos, cube.data[index]), **annotate_kwargs)

    return axis


def plot_labeled_spectrum(spectra_components, spectra_total, line_list, source_location, labels=None, **kwargs):
    """
    Plot a 1D spectra across the MOXSI detector, including components from multiple orders with selected spectral lines
    labeled

    Parameters
    ----------
    spectra_components: `list`
        A list of `ndcube.NDCube` objects containing the spectra for each order component we want to plot.
        Each cube should have a 3D WCS but the data should only have the last dimension of length greater
        than 1.
    spectra_total: `ndcube.NDCube
        The sum of all orders. The shape constraints are the same as the components.
    line_list: `astropy.table.Table`
        A list of all the lines to label. At a minimum, there must be a column labeled "wavelength" and "ion name"
    source_location: `astropy.coordinates.SkyCoord`
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

    fig = plt.figure(figsize=figsize)
    ax = None
    for i, component in enumerate(spectra_components):
        label = labels[i] if labels else None
        if ax is None:
            ax = component[0,0,:].plot(label=label)
        else:
            component[0,0,:].plot(axes=ax, label=label)
        annotate_lines(ax,
                       component[0,0,:],
                       source_location,
                       line_list,
                       component.wcs,
                       cube_total=spectra_total[0,0,:],
                       threshold=threshold,
                       color=ax.lines[-1].get_color(),
                       **kwargs)
    spectra_total[0,0,:].plot(axes=ax, color='k', label='Total')

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
