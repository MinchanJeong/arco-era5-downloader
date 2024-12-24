from pathlib import Path
from datetime import datetime

from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf

def get_config_name():
    config_name = hydra.core.hydra_config.HydraConfig.get().job.config_name
    return config_name

@dataclass
class ARCOERA5Config:
    cfg_name: str
    original_cfg: DictConfig

    gcsfs_object: str
    gcsfs_token: str

    dask_delay: bool
    use_dask_func: bool

    zarr_path: Path

    start_date: datetime
    end_date: datetime
    timestep_hour: int
    shift_forcing: int

    variables: list[str]
    forcing_variables: list[str]

    @classmethod
    def from_omegaconf(cls, args:DictConfig) -> 'ARCOERA5Config':
        return cls(
            cfg_name=get_config_name(),
            original_cfg=args,

            gcsfs_object=str(args.gcsfs.object),
            gcsfs_token=str(args.gcsfs.token),
            
            dask_delay=bool(args.dask.dask_delay),
            use_dask_func=bool(args.dask.use_dask_func),
            
            zarr_path=Path(args.paths.zarr_dir, args.zarr_name),
            
            start_date=datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S'),
            end_date=datetime.strptime(args.end_date, '%Y-%m-%d %H:%M:%S'),
            timestep_hour=args.timestep_hour,
            shift_forcing=args.shift_forcing,
            
            variables=list(args.variables),
            forcing_variables=list(args.forcing_variables)
        )