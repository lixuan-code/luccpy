import os
import xarray as xr
import numpy as np
import sys
from pprint import pprint

from luccpy import calc

demo_y = xr.open_dataarray("demo/demo_y.nc")
demo_x1 = xr.open_dataset("demo/demo_x1.nc").tmp
demo_x2 = xr.open_dataset("demo/demo_x2.nc").spei

data_list = [demo_y,demo_x1,demo_x2]
variables = ["sosyear","tmp","spei"]

pprint(calc.__all__)