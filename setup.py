from setuptools import setup, find_packages

long_description = """
# DGCV - Differential Geometry with Complex Variables

DGCV integrates basic tools for differential geometry with systematic handling of with complex variabless-related structures.

## Tutorials

To get started, check out the Jupyter Notebook tutorials:

- **[DGCV Introduction](https://github.com/YikesItsSykes/DGCV/blob/main/tutorials/DGCV_introduction.ipynb)**: A beginner's guide to setting up and using DGCV.
- **[DGCV in Action](https://github.com/YikesItsSykes/DGCV/blob/main/tutorials/DGCV_in_action.ipynb)**: An advanced tutorial demonstrating DGCV in action.

"""

setup(
    name="DGCV",
    version="0.1.1",
    description="Differential Geometry with Complex Variables",
    long_description=long_description,  # This shows up on PyPI
    long_description_content_type='text/markdown',
    package_dir={"": "src"},  # This tells setuptools that packages are under src/
    packages=find_packages(where="src"),
    package_data={
        'DGCV': ['assets/fonts/*.ttf', 'assets/fonts/fonts.css'],  # Include font files
    },
    include_package_data=True
)


