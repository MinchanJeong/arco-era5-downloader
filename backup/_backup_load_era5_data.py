import os, sys, time, logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import icecream as ic

import xarray as xr
import dask
import dask.array as da

import gcsfs
from dinosaur import xarray_utils

# Setup logging
logging.basicConfig(level=logging.DEBUG)
# direct logging to file
logging.basicConfig(filename='load_era5_data.log', level=logging.DEBUG)

global NGCM_input_variables, NGCM_forcing_variables, precipitation_variable

class DaskManager:
    def __init__(self, dask_delay=False):
        self.dask_delay = dask_delay
        self.delayed_tasks = []

    def process_to_zarr(self, xr_object, **kwargs):
        time.sleep(.5)

        if self.dask_delay and kwargs.get('compute', True) == False:
            delayed_task = da.to_zarr(arr=xr_object, **kwargs)
            self.delayed_tasks.append(delayed_task)
        else:
            task = xr_object.to_zarr(**kwargs)
            dask.compute(task)

    def process_to_zarr_flash(self):
        if self.dask_delay:
            print()
            print("Computing all delayed tasks... LIST:")
            for delayed_task in self.delayed_tasks:
                print(delayed_task)
            dask.compute(*self.delayed_tasks)
            self.delayed_tasks = []
            print("All delayed tasks are computed")

class NeuralGCMGCS:
    def __init__(self, output_dir, start_time, end_time, dask_delay=False):
        self.output_dir = output_dir
        # if output_dir does not exist, create it
        os.makedirs(output_dir, exist_ok=True)
        # lazy loading=
        self.start_time = start_time
        self.end_time = end_time

        self.dask_manager = DaskManager(dask_delay=dask_delay)

    def _set_option(self, option):
        self.option = option
        print(f"Option set to {option}")
        print(f"Output directory set to {self.output_dir}")
        print(f"Setting up ERA5 dataset for {option}")
        self._set_era5_dataset()
        self._get_dataset_info()

    def total_times(self, data_inner_steps):
        return pd.date_range(start=self.start_time, end=self.end_time, freq=f'{data_inner_steps}h')

    def _set_era5_dataset(self):
        gcsfs_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

        if self.option == 'ngcm':
            self.data_inner_steps = 12
            self.target_variables  = NGCM_input_variables + NGCM_forcing_variables
            self.forcing_variables = NGCM_forcing_variables
            self.output_filename = 'NGCM_ERA5_main.zarr'

            self._load_full_era5(gcsfs_path)
            self.sliced_era5 = (
                self.full_era5[self.target_variables]
                .pipe(
                    xarray_utils.selective_temporal_shift,
                    variables=self.forcing_variables,
                    time_shift='24 hours',
                )
                .sel(time=self.total_times(self.data_inner_steps))
            )

        elif self.option == 'gc':
            self.data_inner_steps = 6
            self.target_variables = GC_input_variables
            self.forcing_variables = []
            self.output_filename = 'ERA5_GC.zarr'
            print(f"Loading ERA5 dataset from {gcsfs_path}")
            self._load_full_era5(gcsfs_path)
            print("Slicing ERA5 dataset")
            # total times is defined with start time, end time, and data inner steps as hour
            self.sliced_era5 = (
                self.full_era5[self.target_variables]
                .sel(time=self.total_times(self.data_inner_steps))
                .chunk({'time': 1, 'latitude': -1, 'longitude': -1, 'level': -1})
            )

        elif self.option == 'gc_prec':
            self.data_inner_steps = 1
            self.target_variables = ['total_precipitation']
            self.forcing_variables = []
            self.output_filename = 'ERA5_GC_prec.zarr'
            print(f"Loading ERA5 dataset from {gcsfs_path}")
            self._load_full_era5(gcsfs_path)
            print("Slicing ERA5 dataset")
            self.sliced_era5 = (
                self.full_era5[self.target_variables]
                .sel(time=self.total_times(self.data_inner_steps))
            )

        elif self.option == 'prec':
            self.data_inner_steps = 1
            self.target_variables = precipitation_variables
            self.forcing_variables = []
            self.output_filename = 'ERA5_prec.zarr'
            self._load_full_era5(gcsfs_path)
            self.sliced_era5 = (
                self.full_era5[self.target_variables]
                .sel(time=self.total_times(self.data_inner_steps))
            )
        else:
            raise NotImplementedError("Unsupported option provided: {}".format(self.option))
        

    def _load_full_era5(self, gcsfs_path):
        # https://github.com/google-research/arco-era5
        # https://cloud.google.com/storage/docs/public-datasets/era5
        self.gcs = gcsfs.GCSFileSystem(token='anon')
        gcs_mapper = self.gcs.get_mapper(gcsfs_path)
        self.full_era5 = xr.open_zarr(
            gcs_mapper,
            chunks=None,
            consolidated=True,
        )

    def _get_dataset_info(self):
        self.variables_with_level = {var: ('level' in self.full_era5[var].dims) for var in self.target_variables}
        
        self.coords = self.full_era5.coords
        self.latitude_values = self.coords['latitude'].values
        self.longitude_values = self.coords['longitude'].values
        self.level_values = self.coords['level'].values

    def download_target_era5(self, option):
        self._set_option(option)
        output_path = os.path.join(self.output_dir, self.output_filename)
        print("New Data Download")
        self.process_and_store_data(output_path)
        """
        if os.path.exists(output_path):
            existing_data = xr.open_zarr(output_path)
            self.check_and_fill_missing_data(output_path, existing_data)
        else:
        """
        print("All data downloaded and stored")

    def process_and_store_data(self, output_path):
        # for saving the metadata
        self.sliced_era5.to_zarr(output_path, mode='w', consolidated=True, compute=False)

        print("Storing sample unit time data for metadata")
        self.sliced_era5.sel(time=[self.start_time], drop=False).to_zarr(
            output_path, mode='r+', consolidated=True, compute=True,
            region={'time': slice(0, 1), 'latitude': slice(None), 'longitude': slice(None), 'level': slice(None)}
        )
        print('Storing sample unit time data done')

        print("Downloading and storing data variable-by-variable")
        save_mode = 'r+'
        for var in self.target_variables:
            print(f"Tasking {var}")
            # dask.array.to_zarr() is working with dask.array.
            self.dask_manager.process_to_zarr(self.sliced_era5[var].data, url=output_path, component=var, overwrite=True, compute=False, return_stored=False)
        print("All variables processed and stored")
        self.dask_manager.process_to_zarr_flash()
        
    def check_and_fill_missing_data(self, output_path, existing_data):
        existing_times = existing_data.time.values
        non_existing_times = np.setdiff1d(self.total_times, existing_times)

        if len(non_existing_times) > 0:
            print(f"New data found for {len(non_existing_times)} time steps")
            full_subset_non_existing_times = self.full_era5[self.target_variables].sel(time=non_existing_times)
            new_data = full_subset_non_existing_times.expand_dims('time')
            new_data['time'] = non_existing_times
            self.dask_manager.process_to_zarr(new_data, store=output_path, mode='a', consolidated=True, compute=False, append_dim='time')
            print(f"New data added for {len(non_existing_times)} time steps")

            existing_data = xr.open_zarr(output_path)
            existing_times = existing_data.time.values
        else:
            print("All time steps are already downloaded")

        pbar = tqdm(enumerate(existing_times), total=len(existing_times))
        for time_idx, time in pbar:
            full_subset = self.full_era5.sel(time=time)
            existing_subset = existing_data.sel(time=time)

            for var, with_level in self.variables_with_level.items():
                mask = existing_subset[var].isnull()
                region = {'time': slice(time_idx, time_idx+1),
                            'latitude': slice(None),
                            'longitude': slice(None)}
                if with_level: region['level'] = slice(None)

                if mask.any():
                    full_subset_var = full_subset[var]
                    base_mask = xr.apply_ufunc(np.isnan, full_subset_var)
                    if (mask & ~base_mask).any():

                        pbar.set_description_str(f"Filling missing data for {var}, time: {time}")
                        
                        updated_values = xr.where(mask, full_subset_var, existing_subset[var])

                        # 업데이트된 변수만 포함하는 새로운 데이터셋 생성
                        updated_dataset = existing_subset[var].to_dataset(name=var)
                        updated_dataset[var] = updated_values

                        # 'time' 차원 추가
                        updated_dataset = updated_dataset.expand_dims('time')
                        updated_dataset['time'] = [time]

                        self.dask_manager.process_to_zarr(updated_dataset, store=output_path, mode='a', consolidated=True, compute=False, region=region, group=var)

        self.dask_manager.process_to_zarr_flash()
        return 
    
GC_input_variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    'toa_incident_solar_radiation',
    'temperature',
    'geopotential',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'specific_humidity',
    'total_precipitation',
    'geopotential_at_surface',
    'land_sea_mask',
]

NGCM_input_variables = [
    'geopotential',
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
]

NGCM_forcing_variables = [
    'sea_ice_cover',
    'sea_surface_temperature'
]

precipitation_variables = [
    'convective_rain_rate',
    'large_scale_rain_rate',
    'total_column_rain_water',
    'convective_snowfall',
    'convective_snowfall_rate_water_equivalent',
    'large_scale_snowfall',
    'large_scale_snowfall_rate_water_equivalent',
    'total_precipitation'
]

if __name__ == '__main__':

    output_dir = '/data/minchan/GC_ERA5_0729'
    option = 'gc'  # 'prec' or 'gc' or 'ngcm'
    demo_start_time = '2020-01-01 00:00:00'
    demo_end_time = '2023-01-01 00:00:00'
    dask_delay = True # True Default

    ngcmgcs = NeuralGCMGCS(output_dir, demo_start_time, demo_end_time, dask_delay)
    ngcmgcs.download_target_era5(option)