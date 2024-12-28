from setuptools import setup, find_packages

long_description = """
# DGCV - Differential Geometry with Complex Variables

DGCV integrates tools for differential geometry with systematic handling of complex variables-related structures.

## Tutorials

To get started, check out the Jupyter Notebook tutorials:

- **[DGCV Introduction](https://www.realandimaginary.com/dgcv/tutorials/DGCV_introduction/)**: An introduction to the key concepts and setup.
- **[DGCV in Action](https://www.realandimaginary.com/dgcv/tutorials/DGCV_in_action/)**: A quick tour through examples from some of the library's more elaborate functions.
"""

setup(
    name="DGCV",
    version="0.2.2",
    description="Differential Geometry with Complex Variables",
    long_description=long_description,  # This shows up on PyPI
    long_description_content_type='text/markdown',
    package_dir={"": "src"},  # This tells setuptools that packages are under src/
    packages=find_packages(where="src"),
    package_data={
        'DGCV': ['assets/fonts/*.ttf', 'assets/fonts/fonts.css'],  # Include font files
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'sympy>=1.9',
        'pandas>=1.0',
        'ipython>=7.0'
    ],
)
