import sys
import pandas as pd
from pathlib import Path

import xarray as xr
import dask
import dask.array as da
import gcsfs

import utils
global config
config = utils.get_configs('base')
sys.path.append(config['paths']['repo_path'])
log_path = utils.set_output_path(config)
logger = utils.get_logger('load_era5', log_path, add_stream_handler=True)
logger.info(f"\nConfiguration: {config}\n")

from xarray_utils import selective_temporal_shift

class DaskManager:
    def __init__(self, dask_delay=False):
        self.dask_delay = dask_delay
        self.delayed_tasks = []

    def process_to_zarr(self, da_object, **kwargs):
        arg_compute = kwargs.get('compute', True)
        assert not (arg_compute and self.dask_delay), "compute=True and dask_delay=True are not compatible"

        delayed_task = da.to_zarr(arr=da_object, **kwargs)
        if self.dask_delay:
            self.delayed_tasks.append(delayed_task)

    def process_to_zarr_flash(self):
        if self.dask_delay:
            dask.compute(*self.delayed_tasks)
            self.delayed_tasks = []
            logger.info("All delayed tasks are computed")

class ERA5Downloader:
    def __init__(self, config):
        config_paths = config['paths']
        self.gcsfs_path = config_paths['gcsfs_path']
        self.full_era5 = self._load_full_era5()

        self.start_time = config['start_date']
        self.end_time   = config['end_date']
        self.timestep_hour = config['timestep_hour']
        self.total_times = self._total_times()

        self.dask_manager = DaskManager(dask_delay=config['dask']['dask_delay'])

        self.shift_forcing      = int(config['shift_forcing'])
        self.input_variables   = list(config['variables'])
        self.forcing_variables = list(config['forcing_variables'])
        self.target_variables = self.input_variables + self.forcing_variables
        self.sliced_era5 = self._set_era5_dataset()

        self.zarr_file_path = Path(config_paths['zarr_path'], config_paths['zarr_name'])
    
    def _load_full_era5(self):
        gcs = gcsfs.GCSFileSystem(token='anon')
        gcs_mapper = gcs.get_mapper(self.gcsfs_path)
        return xr.open_zarr(
            gcs_mapper,
            chunks=None,
            consolidated=True,
        )

    def _total_times(self):
        return pd.date_range(start=self.start_time, end=self.end_time, freq=f'{self.timestep_hour}h')

    def _set_era5_dataset(self):
        sliced_era5 = self.full_era5[self.target_variables]

        if self.shift_forcing > 0 and len(self.forcing_variables) > 0:
            sliced_era5 = sliced_era5.pipe(
                selective_temporal_shift,
                variables=self.forcing_variables,
                time_shift=f'{self.shift_forcing} hours',
            )

        sliced_era5 = (
            sliced_era5
            .sel(time=self.total_times)
            .chunk({'time': 1, 'latitude': -1, 'longitude': -1, 'level': -1})
        )

        return sliced_era5

    def _get_dataset_info(self):
        self.variables_with_level    = [var for var in self.target_variables if 'level' in self.sliced_era5[var].dims] 
        self.variables_without_level = [var for var in self.target_variables if 'level' not in self.sliced_era5[var].dims]
        
        self.coords = self.full_era5.coords
        self.latitude_values = self.coords['latitude'].values
        self.longitude_values = self.coords['longitude'].values
        self.level_values = self.coords['level'].values

        logger.info(f"Variables with level: {self.variables_with_level}")
        logger.info(f"Variables without level: {self.variables_without_level}")
        logger.info(f"Latitude values: {self.latitude_values}")
        logger.info(f"Longitude values: {self.longitude_values}")
        logger.info(f"Level values: {self.level_values}")

        logger.info(f"Dataset to be downloaded: {self.sliced_era5}")

        print("Proceed? (y/n)")
        proceed = input()
        if proceed.lower() != 'y':
            sys.exit()

    def process_and_store_data(self):
        self._get_dataset_info()

        # for saving the metadata
        self.sliced_era5.to_zarr(self.zarr_file_path, mode='w', consolidated=True, compute=False)

        logger.info("Storing sample unit time data for metadata")
        self.sliced_era5.sel(time=[self.start_time], drop=False).to_zarr(
            self.zarr_file_path, mode='r+', consolidated=True, compute=True,
            region={'time': slice(0, 1), 'latitude': slice(None), 'longitude': slice(None), 'level': slice(None)}
        )
        logger.info('Storing sample unit time data done')

        logger.info("Downloading and storing data variable-by-variable")
        for var in self.target_variables:
            logger.info(f"Tasking {var}...")
            self.dask_manager.process_to_zarr(self.sliced_era5[var].data, url=self.zarr_file_path, component=var, overwrite=True, compute=False, return_stored=False)
        self.dask_manager.process_to_zarr_flash()
        logger.info("Downloading and storing data done")

if __name__ == '__main__':
    downloader = ERA5Downloader(config)
    downloader.process_and_store_data()