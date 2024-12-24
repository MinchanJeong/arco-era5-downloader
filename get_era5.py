import sys
import pandas as pd
from pathlib import Path
import logging

import xarray as xr
import dask
import dask.array as da
import gcsfs
import zarr
import hydra
from omegaconf import DictConfig

from configs.config import ARCOERA5Config
from utils.logger import set_logger_path, set_logger
from utils.xarray_utils import selective_temporal_shift
from utils.dask_manager import DaskManager
from utils.gcsfs_utils import lazy_load_original_era5


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig) -> None:

    cfg = ARCOERA5Config.from_omegaconf(args)
    logging_path = set_logger_path(cfg)
    set_logger(logging_path)

    total_times = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq=f'{cfg.timestep_hour}h')

    downloader = ERA5Downloader(cfg, total_times)
    downloader._get_dataset_info()

    #print("Proceed? (y/n)")
    #proceed = input()
    #if proceed.lower() != 'y':
    #    sys.exit()
    
    start_time = pd.Timestamp.now()
    downloader.process_and_store_data()
    end_time = pd.Timestamp.now()
    logging.info(f"Total time taken: {end_time - start_time}")

class ERA5Downloader:
    def __init__(self, cfg, total_times):
        self.cfg = cfg
        self.total_times = total_times
        self.full_era5 = lazy_load_original_era5(self.cfg)
        self.sliced_era5 = self._set_era5_dataset()
        self.dask_manager = DaskManager(cfg, self.sliced_era5, self.total_times)

    def _set_era5_dataset(self):
        sliced_era5 = self.full_era5[self.cfg.variables + self.cfg.forcing_variables]

        if self.cfg.shift_forcing > 0 and len(self.cfg.forcing_variables) > 0:
            sliced_era5 = sliced_era5.pipe(
                selective_temporal_shift,
                variables=self.cfg.forcing_variables,
                time_shift=f'{self.cfg.shift_forcing} hours',
            )

        sliced_era5 = (
            sliced_era5
            .sel(time=self.total_times)
            .chunk({'time': 1, 'latitude': -1, 'longitude': -1, 'level': -1})
        )

        return sliced_era5

    def _get_dataset_info(self):
        self.variables_with_level    = [var for var in self.sliced_era5.data_vars if 'level' in self.sliced_era5[var].dims] 
        self.variables_without_level = [var for var in self.sliced_era5.data_vars if 'level' not in self.sliced_era5[var].dims]
        
        coords = self.full_era5.coords
        latitude_values  = coords['latitude'].values
        longitude_values = coords['longitude'].values
        level_values = coords['level'].values

        logging.info(f"Variables with level: {self.variables_with_level}")
        logging.info(f"Variables without level: {self.variables_without_level}")
        logging.info(f"Latitude values: {latitude_values}")
        logging.info(f"Longitude values: {longitude_values}")
        logging.info(f"Level values: {level_values}")

        logging.info(f"Dataset to be downloaded: {self.sliced_era5}")

    def process_and_store_data(self):

        # for saving the metadata
        if not self.cfg.zarr_path.exists():
            self.sliced_era5.to_zarr(self.cfg.zarr_path, mode='w', consolidated=True, compute=False)

        logging.info("Storing sample unit time data for metadata")
        self.sliced_era5.sel(time=[self.cfg.start_date], drop=False).to_zarr(
            self.cfg.zarr_path, mode='r+', consolidated=True, compute=True,
            region={'time': slice(0, 1), 'latitude': slice(None), 'longitude': slice(None), 'level': slice(None)}
        )
        logging.info('Storing sample unit time data done')

        logging.info("Downloading and storing data variable-by-variable")
        for var in self.cfg.variables + self.cfg.forcing_variables:
            logging.info(f"Tasking {var}...")
            if var in self.variables_with_level:
                region_base = {'latitude': slice(None), 'longitude': slice(None), 'level': slice(None)}
            else:
                region_base = {'latitude': slice(None), 'longitude': slice(None)}

            self.dask_manager.process_to_zarr(var, region_base)

            #self.dask_manager.process_to_zarr(self.sliced_era5[var].data, url=self.zarr_file_path, component=var, overwrite=True, compute=False, return_stored=False)
        self.dask_manager.process_to_zarr_flash()
        logging.info("Downloading and storing data done")

if __name__ == '__main__':
    main()