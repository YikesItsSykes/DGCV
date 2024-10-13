from setuptools import setup, find_packages

setup(
    name="DGCV",
    version="0.1.0",
    package_dir={"": "src"},  # This tells setuptools that packages are under src/
    packages=find_packages(where="src"),
    package_data={
        'DGCV': ['assets/fonts/*.ttf', 'assets/fonts/fonts.css'],  # Include font files
    },
    include_package_data=True
)


