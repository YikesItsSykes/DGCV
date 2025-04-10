# **dgcv**: Differential Geometry with Complex Variables

**dgcv** is an open-source Python package providing basic tools for differential geometry integrated with systematic organization of structures naturally accompanying complex variables, in short, Differential Geometry with Complex Variables.

At its core are fully featured symbolic representations of standard DG objects such as vector fields and differential forms, defined relative to coordinate systems that fall into two broad categories:

- **standard** - basic systems that can represent real or complex coordinates, sufficient for all applications that do not require DGCV's complex variable handling features.
- **complex** - richer systems for representing complex coordinate patches, which interact with DGCV's complex variable handling features. These are comprised holomorphic coordinate functions, their conjugates, and their real and imaginary parts (e.g. $\\{z_j,\overline{z_j},x_j,y_j\\}$).

**dgcv** functions account for coordinate types dynamically when oprating on objects built from coordinate systems. The package has a growing library for coordinate-free representations as well, and tools for converting between the two paradigms.

As systems of differential geometric objects constructed from complex variables inherit natural relationships from the underlying complex structure, **dgcv** objects track these relationships across the constructions. Advantages of the uniform integration arise from smooth switching between real and holomorphic coordinate representations of mathematical objects. In computations, **dgcv** objects dynamically manage this format switching on their own so that typical complex variables formulas can be written plainly and will simply work. Some examples of this: In coordinates $z_j = x_j + iy_j$, expressions such as $\frac{\partial}{\partial x_j}|z_j|^2$ or $d z_j \wedge d \overline{z_j} \left( \frac{\partial}{\partial z_j}, \frac{\partial}{\partial y_j} \right)$ are correctly parsed without needing to convert everything to a uniform variable format. Retrieving objects' complex structure-related attributes, like the holomorphic part of a vector field or pluriharmonic terms from a polynomial is straightforward. Complexified cotangent bundles and their exterior algebras are easily decomposed into components from the Dolbeault complex and Dolbeault operators themselves can be applied to functions and k-forms in either coordinate format.

**dgcv** was developed using Python 3.13, and has dependencies on the SymPy and Pandas libraries in addition to base Python.

## Features

- Fully featured symbolic representations of vector fields, differential forms, and tensor fields
- Intuitive interactions with complex structures from holomorphic coordinate systems: **dgcv** objects dynamically manage coordinate transformations between real and holomorphic coordinates during computation as necessary, so objects can be represented in and freely converted between either coordinate format at any time.
- Dedicated python classes for representing common differential geometric structures
- Natural LaTeX rendering for representation of mathematical objects, designed with Jupyter notebooks in mind.

## Installation

You can install **dgcv** directly from PyPI with pip, e.g.:

```bash
pip install dgcv
```

Depending on Python install configurations, the above command can vary. The key is to have the relevant Python environment active so that the package manager pip sources from the right location (suggested to use virtual environments: [Getting started with virtual environments](https://docs.python.org/3/library/venv.html)).

Please note, old versions of **dgcv** where named with capital letters and thus required

```bash
pip install DGCV
```

The upper case name is no longer in use, so this old command will not install the latest version of **dgcv**. The change to using all lower case is to allign with accepted naming standards.

## Tutorials

Two Jupyter Notebook tutorials are available to help getting started with **dgcv**:

1. **dgcv Introduction**: An introduction to the key concepts and setup

   - [View **dgcv** Introduction Tutorial](https://www.realandimaginary.com/dgcv/tutorials/DGCV_introduction/)

2. **dgcv in Action**: A quick tour through examples from some of the library's more elaborate functions
   - [View **dgcv** in Action Tutorial](https://www.realandimaginary.com/dgcv/tutorials/DGCV_in_action/)

### Running the Tutorials Locally

You can download the tutorials individually from the **dgcv** repository [**dgcv** github repo](https://github.com/YikesItsSykes/DGCV).

## Documentation

**dgcv** documentation is hosted at [https://www.realandimaginary.com/dgcv/](https://www.realandimaginary.com/dgcv/), with documentation pages for each function in the library and more. Full documentation is gradually being filled in. In the mean time, docstrings within the code provide more information on available classes/methods and functions.

## License

**dgcv** is licensed under the MIT License. See the `LICENSE.txt` file for more information.

## Author

**dgcv** was created and is maintained by [David Sykes](https://www.realandimaginary.com).

---

### Future Development

The current (0.2.x) version of **dgcv** is foundation on which a lot more can be built. Many additions for future updates are planned, including:

- Extending complex variable handling and dynamic coordinate-type conversion automations. The simple goal is to fully automate handling of complex variable formats, allowing input to be formatted freely with any coordinate type, with features to fully control coordinate type formatting or let the systems automate the process. Much more will be added to this end.
- Expanding libraries dedicated to more specialized areas of differential geometry

Contributions and feedback from anyone interested are warmly welcomed.
Stay tuned for more updates!
