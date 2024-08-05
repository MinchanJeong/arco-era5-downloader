import numpy as np
import pandas as pd
import xarray

from typing import Sequence


def selective_temporal_shift(
    dataset: xarray.Dataset,
    variables: Sequence[str] = tuple(),
    time_shift: str|np.timedelta64|pd.Timedelta = '0 hour',
    time_name: str = 'time',
) -> xarray.Dataset:
  """Shifts specified variables in time and truncates associated time values.

  As with `xarray.Dataset.shift()`, positive values of the shift move values
  "to the right", negative values "to the left" relative to the original dataset
  time coordinates. This implies that specifying a positive `time_shift` will
  produce a dataset where for each time the values of `variables` are from an
  earlier time in the original dataset. See unit tests for examples.

  Args:
    dataset: Input dataset.
    variables: Variables to which shift is applied.
    time_shift: Timedelta by which to shift `variables.`
    time_name: Name of the time coordinate.

  Returns:
    Dataset where every DataArray in `variables` have been shifted on the
    `time_name` coordinate by `time_shift`.  The head or tail times associated
    with the shifted indices are truncated.
  """
  time_shift = pd.Timedelta(time_shift)
  time_spacing = dataset[time_name][1] - dataset[time_name][0]

  shift, remainder = divmod(time_shift, time_spacing)
  shift = int(shift)  # convert from xarray value
  if shift == 0 or not variables:
    return dataset
  if remainder:
    raise ValueError(f'Does not divide evenly, got {remainder=}')

  ds = dataset.copy()
  if shift > 0:
    ds = ds.isel({time_name: slice(shift, None)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(None, -shift)})
  else:
    ds = ds.isel({time_name: slice(None, shift)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(-shift, None)})
  return ds