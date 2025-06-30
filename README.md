# **dgcv**: Differential Geometry with Complex Variables

**dgcv** is an open-source Python package providing basic tools for differential geometry integrated with systematic organization of structures naturally accompanying complex variables, in short, Differential Geometry with Complex Variables.

At its core are symbolic representations of standard DG objects such as vector fields and differential forms. There are coordinate free representations, and representations defined relative to coordinate systems falling into two general categories:

- **standard** - basic systems that can represent real or complex coordinates, sufficient for applications that do not require **dgcv**'s complex variable handling features.
- **complex** - richer systems for representing complex coordinate patches that interact with **dgcv**'s complex variable handling features. These are comprised of holomorphic coordinate functions, their conjugates, and their real and imaginary parts (e.g., $\\{z_j,\overline{z_j},x_j,y_j\\}$).

**dgcv** functions account for coordinate types dynamically when operating on objects built from such coordinate systems. The package has a growing library for coordinate-free representations as well, and tools for converting between the two paradigms.

As systems of differential geometric objects constructed from complex variables inherit natural relationships from the underlying complex structure, **dgcv** objects track these relationships across the constructions. This system enables smooth switching between real and holomorphic coordinate representations of mathematical objects. In computations, **dgcv** objects dynamically manage this format switching on their own so that typical complex variables formulas can be written plainly and will simply work. Some examples of this: In coordinates $z_j = x_j + iy_j$, expressions such as $\frac{\partial}{\partial x_j}|z_j|^2$ or $d z_j \wedge d \overline{z_j} \left( \frac{\partial}{\partial z_j}, \frac{\partial}{\partial y_j} \right)$ are correctly parsed without needing to convert everything to a uniform variable format. Retrieving objects' complex structure-related attributes, like the holomorphic part of a vector field or pluriharmonic terms from a polynomial is straightforward. Complexified cotangent bundles and their exterior algebras are easily decomposed into components from the Dolbeault complex and Dolbeault operators themselves can be applied to functions and k-forms in either coordinate format.

**dgcv** is tested on Python 3.13, and has dependencies on the SymPy, Pandas, and iPython libraries.

## Features

- Symbolic representations of various tensor fields (vector fields, differential forms, etc.) and dedicated python classes for representing many other common differential geometric structures
- Intuitive interactions with complex structures from holomorphic coordinate systems: **dgcv** objects dynamically manage coordinate transformations between real and holomorphic coordinates during computation as necessary, so objects can be represented in and freely converted between either coordinate format

## Installation

**dgcv** can be installed directly from PyPI with pip, e.g.:

```bash
pip install dgcv
```

Depending on Python install configurations, the above command can vary. The key is to have the relevant Python environment active so that the package manager `pip` sources from the right location (suggested to use virtual environments: [Getting started with virtual environments](https://docs.python.org/3/library/venv.html)).

## Documentation

**dgcv** documentation is hosted at [https://www.realandimaginary.com/dgcv/](https://www.realandimaginary.com/dgcv/), with documentation pages for individual functions in the library and more. Docstrings within the code provide more information on available classes/methods and functions.

## License

**dgcv** is licensed under the MIT License. See the [`LICENSE.txt`](https://github.com/YikesItsSykes/dgcv/blob/main/LICENSE.txt) file for more information.

## Author

**dgcv** was created and is maintained by [David Sykes](https://www.realandimaginary.com).

---

### Future Development

Version 0.3.6 of **dgcv** is designed with several planned extensions to the library in mind. Current plans broadly fall into two categories:

- Extending complex variable handling and dynamic coordinate-type conversion automations. The simple goal is to fully automate handling of complex variable formats, allowing input to be formatted freely with any coordinate type, with features to fully control coordinate type formatting or let the systems automate the process.
- Expanding libraries dedicated to more specialized areas of differential geometry

Contributions, requests for additions to the library, and feedback from anyone interested are very much welcome.
