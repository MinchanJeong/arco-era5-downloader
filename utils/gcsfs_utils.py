# Description: Utility functions for interacting with GCSFS
import gcsfs
import xarray as xr

def lazy_load_original_era5(cfg):
    gcs = gcsfs.GCSFileSystem(token=cfg.gcsfs_token)
    gcsfs_path = cfg.gcsfs_object
    full_era5 = xr.open_zarr(gcs.get_mapper(gcsfs_path), chunks=None, consolidated=None)
    return full_era5