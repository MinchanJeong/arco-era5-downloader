import gcsfs
import xarray as xr

gcs = gcsfs.GCSFileSystem(token='anon')
path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'
full_era5 = xr.open_zarr(gcs.get_mapper(path), chunks=None)
print(full_era5)

for var in full_era5:
    print(var)
