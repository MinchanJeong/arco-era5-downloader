import os
import dask
import xarray as xr
import gcsfs
import numpy as np
from tqdm import tqdm
from dinosaur import xarray_utils
from dask import delayed, compute

input_variables = [
    'geopotential',
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
]

forcing_variables = [
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

gcs = gcsfs.GCSFileSystem(token='anon')

def initialize_gcs_and_era5(path):
    full_era5 = xr.open_zarr(gcs.get_mapper(path), chunks=None)
    return full_era5

def get_target_variables(option):
    if option == 'neuralgcm':
        data_inner_steps = 12
        target_variables = input_variables + forcing_variables
        output_filename = 'NGCM_ERA5_main.zarr'
    elif option == 'precipitation':
        data_inner_steps = 1
        target_variables = precipitation_variables
        output_filename = 'NGCM_ERA5_prec.zarr'
    else:
        raise NotImplementedError("Unsupported option provided: {}".format(option))

    return target_variables, data_inner_steps, output_filename

def get_forcing_variables(option):
    if option == 'neuralgcm':
        return forcing_variables
    else:
        return []


def check_and_fill_missing_data(full_era5, existing_data, target_variables, total_times, output_path):
    variables_with_level = {var: ('level' in full_era5[var].dims) for var in target_variables}
    
    metadata = full_era5.coords
    latitude_values = metadata['latitude'].values
    longitude_values = metadata['longitude'].values
    level_values = metadata['level'].values
    print(f"Latitude: {latitude_values}")
    print(f"Longitude: {longitude_values}")
    print(f"Level: {level_values}")

    pbar = tqdm(total_times, desc="Checking missing data regions")

    for time in pbar:
        time_idx = np.where(total_times == time)[0][0]
        full_subset = full_era5.sel(time=time)

        if np.datetime64(time) not in existing_data.time.values:
            pbar.set_description_str(f"Adding data for time: {time}")
            # If the time is not in existing_data, add the data from full_era5
            new_data = full_subset[target_variables].expand_dims('time')
            new_data['time'] = [time]

            with dask.config.set(scheduler='threads'):
                store_task = new_data.to_zarr(output_path, mode='a', consolidated=True, compute=False, append_dim='time')
                dask.compute(store_task)

        else:
            existing_subset = existing_data.sel(time=time)
            time_idx = np.where(existing_data.time.values == np.datetime64(time))[0][0]
            
            for var, with_level in variables_with_level.items():
                mask = existing_subset[var].isnull()

                region = {'time': slice(time_idx, time_idx+1),
                            'latitude': slice(None),
                            'longitude': slice(None)}
                if with_level: region['level'] = slice(None)

                if mask.any():
                    pbar.set_description_str(f"Filling missing data for {var}, time: {time}")
                    full_subset_var = full_subset[var]
                    updated_values = xr.where(mask, full_subset_var, existing_subset[var])

                    # 업데이트된 변수만 포함하는 새로운 데이터셋 생성
                    updated_dataset = existing_subset[var].to_dataset(name=var)
                    updated_dataset[var] = updated_values

                    # 'time' 차원 추가
                    updated_dataset = updated_dataset.expand_dims('time')
                    updated_dataset['time'] = [time]

                    with dask.config.set(scheduler='threads'):
                        store_task = updated_dataset.to_zarr(output_path, mode='a', consolidated=True, compute=False, region=region)
                        dask.compute(store_task)

    return 

def process_and_store_data(full_era5, target_variables, forcing_variables,\
                            demo_start_time, demo_end_time, data_inner_steps, output_path, option,\
                            download_metadata_only):
    if option == 'neuralgcm':
        sliced_era5 = (
            full_era5[target_variables]
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=forcing_variables,
                time_shift='24 hours',
            )
        )
    else:
        sliced_era5 = (
            full_era5[target_variables]
        )

    print(sliced_era5)

    if download_metadata_only:
        pass

    with dask.config.set(scheduler='threads'):
        store_task = sliced_era5.chunk({'time': 100}).to_zarr(output_path, mode='w', consolidated=True, compute=False)

    dask.compute(store_task)

    return

if __name__ == '__main__':
    option = 'neuralgcm'  # 또는 'precipitation'
    path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
    output_dir = '/data/minchan/NGCM_ERA5_ver2'
    demo_start_time = '2020-01-01 00:00:00'
    demo_end_time = '2023-01-01 00:00:00'

    download_metadata_only = True

    full_era5 = initialize_gcs_and_era5(path)

    target_variables, data_inner_steps, output_filename = get_target_variables(option)
    forcing_variables = get_forcing_variables(option)
    output_path = os.path.join(output_dir, output_filename)

    total_times = np.arange(np.datetime64(demo_start_time), np.datetime64(demo_end_time) + np.timedelta64(1, 'ns'), np.timedelta64(data_inner_steps, 'h'))
    full_era5 = full_era5.sel(time=total_times)

    if os.path.exists(output_path):
        existing_data = xr.open_zarr(output_path)
        print(existing_data)
        print(existing_data.time)
        check_and_fill_missing_data(full_era5, existing_data, target_variables, total_times, output_path)
    else:
        process_and_store_data(full_era5, target_variables, forcing_variables,\
                                demo_start_time, demo_end_time, data_inner_steps, output_path, option,\
                                      download_metadata_only)
