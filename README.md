### Description
Personal ERA5 Downloader

### USAGE
`python3 get_era5.py --config-name=graphcast`

### Requirements
gcsfs, xarray, zarr, dask

### Dask Reference
- https://examples.dask.org/xarray.html
- dask.delayed: https://docs.dask.org/en/stable/delayed-best-practices.html
    - minimize dask.compute() calls
    - https://jjongguet.tistory.com/123
- da.to_zarr: https://docs.dask.org/en/stable/generated/dask.array.to_zarr.html
- da.store: https://docs.dask.org/en/latest/generated/dask.array.store.html


### ERA5
- https://github.com/google-research/arco-era5
- https://cloud.google.com/storage/docs/public-datasets/era5


### Further Development
- Study cloud optimization strategy