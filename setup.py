from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.0'

setup(
    name='luccpy',  # package name
    version=VERSION,  # package version
    description='LUCC & Climate Group for Personal Use',  # package description
    url="https://github.com/lixuan-code/",
    author="xfry",
    zip_safe=False,
    keywords=[
        "Python",
    ],
    install_requires=[
        "dask",
        "anyio==3.*",
        'numpy==1.26.4',
        'pandas==2.2.2',
        'xarray==2023.7.0',
        'rioxarray==0.15.0',
        'scipy==1.13.0',
        'pingouin==0.5.4',
        'shap==0.45.0',
        'salem==0.3.10',
        'scikit-learn==1.4.2',
        'pymannkendall==1.4.3',
        'netcdf4==1.6.5',
        "geocube==0.4.2",
        "geopandas==0.14.4",
        "matplotlib==3.8.4",
        "jupyter==1.0.0",
        "ipython==8.18.1",
        "ipykernel==6.29.3"
    ],
    python_requires=">=3.9",
    packages=find_packages("src"),
    package_dir={"": "src"},

)

