"""
Script to compute uncertainty
"""
import copy
import pathlib

import astropy.units as u
import numpy as np
import sunpy.map


if __name__ == '__main__':
    m_l1 = sunpy.map.Map(snakemake.input[0])
    m_l2 = sunpy.map.Map(snakemake.input[1])
    # NOTE: need to know approximately how many pixels we combined into each new pixel
    # in order to compute the error on each map in the case of the AIA images
    area_1 = m_l1.scale.axis1 * m_l1.scale.axis2
    area_2 = m_l2.scale.axis1 * m_l2.scale.axis2
    n_sample = int(np.round((area_2 / area_1).decompose()).value)
    percent_error = float(snakemake.config['percent_error']) * u.percent
    error = percent_error * m_l2.quantity / np.sqrt(n_sample)
    new_meta = copy.deepcopy(m_l2.meta)
    new_meta['measrmnt'] = 'uncertainty'
    error_map = m_l2._new_instance(error.to_value(m_l2.unit), meta=new_meta)
    output_dir = pathlib.Path(snakemake.output[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    error_map.save(snakemake.output[0])
