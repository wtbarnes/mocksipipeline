"""
Client for searching synoptic XRT data from MSU
"""
import os
import warnings

import parse

from sunpy.net import attrs as a
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever import GenericClient
from sunpy.net.scraper import Scraper
from sunpy.time import TimeRange

from xrtpy.response.effective_area import index_mapping_to_fw1_name, index_mapping_to_fw2_name


__all__ = ['XRTSynopticClient', 'FilterWheel1', 'FilterWheel2']


# invert mapping
fw1_name_to_index_mapping = {v:k for k,v in index_mapping_to_fw1_name.items()}
fw2_name_to_index_mapping = {v:k for k,v in index_mapping_to_fw2_name.items()}


class FilterWheel1(SimpleAttr):
    
    def __init__(self, fw_1_name):
        """
        Names of the filters on filter wheel 1 and filter wheel 2
        """
        fw_1_name = fw_1_name.strip().lower().capitalize()
        if fw_1_name not in index_mapping_to_fw1_name:
            raise IndexError(f'{fw_1_name} is not a valid choice for filter wheel 1. '
                             f'Allowed values are {index_mapping_to_fw1_name.keys()}')
        fw_1_int = index_mapping_to_fw1_name[fw_1_name]

        super().__init__(f'{fw_1_int}')


class FilterWheel2(SimpleAttr):
    
    def __init__(self, fw_2_name):
        """
        Names of the filters on filter wheel 1 and filter wheel 2
        """
        fw_2_name = fw_2_name.strip().lower().capitalize()
        if fw_2_name not in index_mapping_to_fw2_name:
            raise IndexError(f'{fw_2_name} is not a valid choice for filter wheel 2. '
                             f'Allowed values are {index_mapping_to_fw2_name.keys()}')
        fw_2_int = index_mapping_to_fw2_name[fw_2_name]

        super().__init__(f'{fw_2_int}')


class XRTSynopticClient(GenericClient):
    """
    Search synoptic XRT images provided by MSU
    """
    baseurl = r'http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_images/syncmp_PNG/%Y/%m/SYN_XRT%Y%m%d_%H%M%S.(\d){1}.1024.(\d){2}.png'
    pattern = r'{}/{year:4d}/{month:2d}/SYN_XRT{year:4d}{month:2d}{day:2d}_{hour:2d}{minute:2d}{second:2d}.{:1d}.1024.{FilterWheel1}{FilterWheel2}.png'

    @property
    def info_url(self):
        return 'http://solar.physics.montana.edu/HINODE/XRT/SCIA/about_scia.html'

    @classmethod
    def register_values(cls):
        return {
            a.Instrument: [('XRT', 'X-ray Telescope')],
            a.Physobs: [('intensity', '')],
            a.Source: [('Hinode', 'The Hinode mission')],
            a.Provider: [('MSU', 'Montana State University')],
            a.Level: [('2', 'synoptic images are level 2 files')],
            FilterWheel1: [(str(f), f'Filter wheel 1: {f}') for f in fw1_name_to_index_mapping.keys()],
            FilterWheel2: [(str(f), f'Filter wheel 2: {f}') for f in fw1_name_to_index_mapping.keys()],
        }

    def _scrape_fits_files_for_timerange(self, matchdict):
        # TODO: add comment about why we have to do this
        fits_base_url = r'http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/'
        fits_pattern = r'%Y/%m/%d/H%H00/comp_XRT%Y%m%d_%H%M%S.(\d){1}.fits'
        fits_scraper = Scraper(fits_base_url+fits_pattern, regex=True)
        fits_urls = fits_scraper.filelist(TimeRange(matchdict['Start Time'], b=matchdict['End Time']))
        return {os.path.basename(f): f for f in fits_urls}

    def _get_key_from_png_url(self, url):
        p = parse.parse('SYN_XRT{d1}_{d2}.{d3}.1024.{}.png', os.path.basename(url))
        return f"comp_XRT{p.named['d1']}_{p.named['d2']}.{p.named['d3']}.fits"

    def post_search_hook(self, exdict, matchdict):
        rd = super().post_search_hook(exdict, matchdict)
        # TODO: scrape the corresponding FITS URLs
        # Replace the png URL with the FITS url for each result
        if not hasattr(self, 'fits_mapping'):
            self.fits_mapping = self._scrape_fits_files_for_timerange(matchdict)
        fits_key = self._get_key_from_png_url(rd['url'])
        rd['url'] = self.fits_mapping.get(fits_key, None)
        # Replace filter wheel index with name of filter wheels
        index_1 = rd['FilterWheel1']
        index_2 = rd['FilterWheel2']
        del rd['FilterWheel1']
        del rd['FilterWheel2']
        rd['Filter Wheel 1'] = fw1_name_to_index_mapping[int(index_1)]
        rd['Filter Wheel 2'] = fw2_name_to_index_mapping[int(index_2)]
        return rd

    def search(self, *args, **kwargs):
        # Drop rows where the URL is missing because there is not a corresponding FITS
        # file
        q = super().search(*args, **kwargs)
        del_rows = []
        for i,r in enumerate(q):
            if r['url'] is None:
                warnings.warn(f'No FITS file available for {r}. Dropping row.')
                del_rows.append(i)
        q.remove_rows(del_rows)
        return q

    @classmethod
    def _can_handle_query(cls, *query):
        required = {a.Time, a.Instrument, a.Source}
        optional = {a.Provider, a.Physobs, a.Level, FilterWheel1, FilterWheel2}
        return cls.check_attr_types_in_query(query, required, optional)
