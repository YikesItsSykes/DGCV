DGCV is an open-source Python package for differential geometry with complex variables. It provides symbolic representations of standard DG objects like vector fields and differential forms in complex coordinate systems. DGCV tracks natural relationships between complex geometric objects constructed from complex variables. This uniform integration allows smooth switching between real and holomorphic coordinate representations. DGCV classes dynamically manage format switching, so complex variables formulas work without conversion. For example, expressions like $\frac{\partial}{\partial x_j}|z_j|^2$ or $d z_j \wedge d \overline{z_j} \left( \frac{\partial}{\partial z_j}, \frac{\partial}{\partial y_j} \right)$ are parsed correctly. Retrieving complex structure-related attributes, like the holomorphic part of a vector field or pluriharmonic terms from a polynomial, is straightforward. Complexified cotangent bundles and their exterior algebras can be decomposed into components from the Dolbeault complex and Dolbeault operators applied to functions and k-forms in either coordinate format.

DGCV, developed with Python 3.12, integrates SymPy objects within its data structures and subclasses from SymPy.Basic, inheriting much of SymPy’s functionality. It supports symbolic representations of vector fields, differential forms, and tensor fields. DGCV objects dynamically manage coordinate transformations between real and holomorphic coordinates during computation. It also has dedicated classes for representing common differential geometric structures like Riemannian metrics and Kahler structures. Custom LaTeX rendering provides a clean visual representation of mathematical objects, ideal for sharing ideas in Jupyter notebooks.

Install DGCV directly from PyPI with pip:
```bash
pip install DGCV
```

Note that on macOS/Linux, use `python3` before `pip`, while on Windows, use `py`.

For more information, visit the DGCV documentation.

Two Jupyter Notebook tutorials are available to help with DGCV:

- **DGCV Introduction:** An introduction to key concepts and setup. [View Tutorial](https://www.realandimaginary.com/dgcv/tutorials/DGCV_introduction/)
- **DGCV in Action:** A quick tour through examples from the library’s functions. [View Tutorial](https://www.realandimaginary.com/dgcv/tutorials/DGCV_in_action/)

To run the tutorials locally, clone the DGCV GitHub repository and navigate to the tutorials folder. Use Jupyter to open the corresponding notebooks. Alternatively, download the tutorials from the repository.

DGCV documentation is hosted at [https://www.realandimaginary.com/dgcv/](https://www.realandimaginary.com/dgcv/), with documentation for each library function. Full documentation is under development, so refer to the docstrings for more information.

DGCV is licensed under the MIT License. See the `LICENSE.txt` file for details.

David Sykes created and maintains DGCV.
The current (0.x.x) version of DGCV serves as a foundation for future updates. Planned additions include:

- Extending complex variable handling and dynamic coordinate-type conversion automations. This aims to fully automate handling of complex variable formats, allowing input with any coordinate type and control over formatting. The current API meets this goal for core classes but not ancillary classes, such as differential forms and general tensor fields.
- Expanding libraries for specialized areas of differential geometry, including Symplect/Contact Hamiltonian formalism, CR structures, Riemannian and Kahler, Sasakian, etc.

Contributions and feedback are welcome.
Stay tuned for more updates!
