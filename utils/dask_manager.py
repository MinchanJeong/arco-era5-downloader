import xarray as xr
import dask
import dask.array as da
import logging

class DaskManager:

    def __init__(self, cfg, sliced_era5, total_times):
        self.cfg = cfg
        self.sliced_era5 = sliced_era5
        self.total_times = total_times

        self.dask_delay = self.cfg.dask_delay
        self.zarr_path  = self.cfg.zarr_path
        
        if self.cfg.use_dask_func:
            self.process_to_zarr = self.process_to_zarr_by_dask
        else:
            self.process_to_zarr = self.process_to_zarr_by_xarray

        self.delayed_tasks = []

    def process_to_zarr_by_xarray(self, var, region_base):
        for time_idx, time in enumerate(self.total_times):
            if time_idx == 0: continue
            region = region_base.copy()
            region['time'] = slice(time_idx, time_idx+1)
            nullspace = xr.open_zarr(self.zarr_file_path, consolidated=True).isnull()
            if nullspace[var].sel(time=[time], drop=False).any():
                dask_delay = self.sliced_era5[var].sel(time=[time], drop=False).to_zarr(self.zarr_file_path, mode='r+', consolidated=True, compute=False, region=region)
            
                if self.dask_delay:
                    self.delayed_tasks.append(dask_delay)

    def process_to_zarr_by_dask(self, var, region_base):
        delayed_task = da.to_zarr(arr=self.sliced_era5[var].data, \
                                  url=self.zarr_path, component=var, overwrite=True, compute=False, return_stored=False)
        if self.dask_delay:
            self.delayed_tasks.append(delayed_task)

    def process_to_zarr_flash(self):
        if self.dask_delay:
            logging.info("Computing all delayed tasks... Setting Logger level to DEBUG for more details")
            logging.getLogger().setLevel(logging.DEBUG)
            dask.compute(*self.delayed_tasks)
            self.delayed_tasks = []
            logging.info("All delayed tasks are computed")