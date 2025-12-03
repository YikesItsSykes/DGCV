from setuptools import find_packages, setup

long_description = """
# dgcv - Differential Geometry with Complex Variables

dgcv integrates tools for differential geometry with systematic handling of complex variables-related structures.

## Tutorials

To get started, check out the Jupyter Notebook tutorials:

- **[dgcv Introduction](https://www.realandimaginary.com/dgcv/tutorials/dgcv_introduction/)**: An introduction to the key concepts and setup.
- **[dgcv in Action](https://www.realandimaginary.com/dgcv/tutorials/dgcv_in_action/)**: A quick tour through examples from some of the library's more elaborate functions.
"""

setup(
    name="dgcv",
    version="0.3.12",
    description="Differential Geometry with Complex Variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "dgcv": ["assets/fonts/*.ttf", "assets/fonts/fonts.css"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["sympy>=1.9", "ipython>=7.0"],
)
