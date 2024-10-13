import re
from pandas import DataFrame, MultiIndex
from .config import get_variable_registry, greek_letters
from .styles import get_style


# Modular function to collect data for complex, standard, and algebra variables
def collect_variable_data(variable_registry, use_latex):
    data = [['Number of Var.', 'Real Part', 'Imaginary Part', 'Vector Fields', 'Differential Forms']]
    index = []

    complex_vars = sorted(variable_registry.get('complex_variable_systems', {}).keys())
    standard_vars = sorted(variable_registry.get('standard_variable_systems', {}).keys())
    finite_algebra_vars = sorted(variable_registry.get('finite_algebra_systems', {}).keys())

    # Process complex variables
    for var_name in complex_vars:
        process_variable(var_name, 'complex', variable_registry, data, index, use_latex)

    # Process standard variables
    for var_name in standard_vars:
        process_variable(var_name, 'standard', variable_registry, data, index, use_latex)

    # Process finite algebra systems
    for var_name in finite_algebra_vars:
        process_algebra(var_name, variable_registry, data, index, use_latex)

    return data, index


# Function to process complex and standard variables
def process_variable(var_name, system_type, variable_registry, data, index, use_latex):
    # Common logic for both standard and complex variable systems
    family_type = variable_registry[f'{system_type}_variable_systems'][var_name].get('family_type', 'single')
    family_names = variable_registry[f'{system_type}_variable_systems'][var_name]['family_names']
    initial_index = variable_registry[f'{system_type}_variable_systems'][var_name].get('initial_index', 1)
    
    # Logic to process tuple or single variables and append to data
    tuple_len = len(family_names)
    real_part, imaginary_part, vf_str, df_str = format_variable_details(var_name, family_names, initial_index, tuple_len, system_type, use_latex)
    
    # Append to data
    data.append([
        tuple_len,
        real_part,
        imaginary_part,
        vf_str,
        df_str
    ])

    # Update the index
    formatted_name = format_variable_name(var_name, system_type, use_latex)
    index.append((formatted_name, ''))


# Function to process algebra systems
def process_algebra(var_name, variable_registry, data, index, use_latex):
    system_data = variable_registry['finite_algebra_systems'][var_name]
    algebra_family_names = system_data.get('family_names', [])
    
    # Format algebra label and basis
    formatted_str = format_algebra_name(var_name, algebra_family_names, use_latex)
    
    # Append to data
    data.append([len(algebra_family_names), '----', '----', '----', '----'])
    
    # Update the index
    index.append((formatted_str, ''))


# Function to build the table
def build_table(data, index, style):
    # Build and return the final styled table
    columns = MultiIndex.from_product([['Initialized Coordinate Systems and Algebras'], ['', '', '', '', '']])
    table = DataFrame(data=data, index=index, columns=columns)

    # Apply styles and return the table
    table_styles = get_style(style)
    return table.style.set_table_styles(table_styles)

# Helper function to format variable details (real part, imaginary part, vector fields, differential forms)
def format_variable_details(var_name, family_names, initial_index, tuple_len, system_type, use_latex):
    if system_type == 'complex':
        variable_registry=get_variable_registry()
        # Access the real and imaginary parts (assuming they are the 3rd and 4th elements in family_names)
        family_houses = variable_registry['complex_variable_systems'][var_name]['family_houses']
        family_names = variable_registry['complex_variable_systems'][var_name]['family_names']

        if len(family_names[2]) > 1:
            real_part = f"{family_houses[2]} = ({family_names[2][0]}, ..., {family_names[2][-1]})"
        else:
            real_part = f"{family_houses[2]} = ({family_names[2][0]})"

        if len(family_names[3]) > 1:
            imaginary_part = f"{family_houses[3]} = ({family_names[3][0]}, ..., {family_names[3][-1]})"
        else:
            imaginary_part = f"{family_houses[3]} = ({family_names[3][0]})"

        vf_str = build_object_string_for_complex('D', family_houses, family_names, initial_index, use_latex)
        df_str = build_object_string_for_complex('d', family_houses, family_names, initial_index, use_latex)
        
    else:  # Standard variables
        real_part = '----'  # Real part not applicable for standard variables
        imaginary_part = '----'  # Imaginary part not applicable for standard variables
        vf_str = build_object_string('D', var_name, initial_index, tuple_len, system_type, use_latex)
        df_str = build_object_string('d', var_name, initial_index, tuple_len, system_type, use_latex)

    return real_part, imaginary_part, vf_str, df_str

def format_variable_name(var_name, system_type, use_latex=False):
    variable_registry=get_variable_registry()
    if system_type == 'standard':
        family_type = variable_registry['standard_variable_systems'][var_name].get('family_type', 'single')
        family_names = variable_registry['standard_variable_systems'][var_name]['family_names']
        initial_index = variable_registry['standard_variable_systems'][var_name].get('initial_index', 1)
    else:  # system_type == 'complex'
        family_type = variable_registry['complex_variable_systems'][var_name].get('family_type', 'single')
        family_names = variable_registry['complex_variable_systems'][var_name]['family_names'][0]  # First tuple entry
        initial_index = variable_registry['complex_variable_systems'][var_name].get('initial_index', 1)

    # Handle tuple or single variable cases
    if family_type == 'tuple':
        first_index = initial_index
        last_index = initial_index + len(family_names) - 1
        if use_latex:
            content = f"{convert_to_greek(var_name)} = \\left( {convert_to_greek(var_name)}_{{{first_index}}}, \\ldots, {convert_to_greek(var_name)}_{{{last_index}}} \\right)"
        else:
            content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
    else:
        content = convert_to_greek(var_name) if use_latex else var_name

    return wrap_in_dollars(content) if use_latex else content

def format_algebra_name(var_name, algebra_family_names, use_latex=False):
    if use_latex:
        match = re.match(r'(.*?)(\d+)?$', var_name)
        algebra_name = match.group(1) if match else var_name
        algebra_number = match.group(2) if match else None

        # Format the algebra name in LaTeX
        algebra_label = f"\\mathfrak{{{convert_to_greek(algebra_name)}}}" if re.fullmatch(r'[a-z]+', algebra_name) else algebra_name
        if algebra_number:
            algebra_label += f"_{{{algebra_number}}}"

        # Format the basis string
        if len(algebra_family_names) > 1:
            first_basis_label = process_basis_label(algebra_family_names[0])
            last_basis_label = process_basis_label(algebra_family_names[-1])
            basis_str = f"{first_basis_label}, \\ldots, {last_basis_label}"
        else:
            basis_str = process_basis_label(algebra_family_names[0]) if algebra_family_names else ""
        return f"Algebra: {wrap_in_dollars(algebra_label)} \\\\ Basis: {wrap_in_dollars(basis_str)}"
    else:
        algebra_label = var_name
        if len(algebra_family_names) > 1:
            basis_str = f"{algebra_family_names[0]}, ..., {algebra_family_names[-1]}"
        else:
            basis_str = algebra_family_names[0] if algebra_family_names else ""
        return f"Algebra: {algebra_label}<br>Basis: {basis_str}"
    
def build_object_string_for_complex(obj_type, part_names, family_names, start_index, use_latex=False):
    parts = []
    for part_name, part in zip(part_names, family_names):
        if use_latex:
            base_var = convert_to_greek(part_name.replace('BAR', '', 1)) if part_name.startswith('BAR') else convert_to_greek(part_name)
            if len(part) > 1:
                part_str = f"\\overline{{{base_var}_{{{start_index}}}}}, \\ldots, \\overline{{{base_var}_{{{start_index + len(part) - 1}}}}}"
            else:
                part_str = f"\\overline{{{base_var}}}" if part_name.startswith('BAR') else base_var
            parts.append(f"d {part_str}" if obj_type == 'd' else f"\\frac{{\\partial}}{{\\partial {part_str}}}")
        else:
            part_str = f"{obj_type}_{convert_to_greek(part_name)}{start_index},...,{convert_to_greek(part_name)}{start_index + len(part) - 1}" if len(part) > 1 else f"{obj_type}_{convert_to_greek(part_name)}"
            parts.append(part_str)
    return ', '.join(parts)

def build_object_string(obj_type, var_name, start_index, tuple_len, system_type, use_latex=False):
    if tuple_len == 1:
        if use_latex:
            content = f"d {convert_to_greek(var_name)}" if obj_type == 'd' else f"\\frac{{\\partial}}{{\\partial {convert_to_greek(var_name)}}}"
        else:
            content = f"{obj_type}_{var_name}"
    else:
        if use_latex:
            content = f"d {convert_to_greek(var_name)}_{{{start_index}}}, \\ldots, d {convert_to_greek(var_name)}_{{{start_index + tuple_len - 1}}}" if obj_type == 'd' else f"\\frac{{\\partial}}{{\\partial {convert_to_greek(var_name)}_{{{start_index}}}}}, \\ldots, \\frac{{\\partial}}{{\\partial {convert_to_greek(var_name)}_{{{start_index + tuple_len - 1}}}}}"
        else:
            content = f"{obj_type}_{var_name}{start_index},...,{obj_type}_{var_name}{start_index + tuple_len - 1}"
    return wrap_in_dollars(content) if use_latex else content

# Helper function to convert variable names to Greek letters (if applicable)
def convert_to_greek(var_name):    
    # Replace variable names with their corresponding Greek letters
    for name, greek in greek_letters.items():
        if var_name == name:
            return greek
    return var_name

# Helper function to wrap content in LaTeX dollar signs for rendering
def wrap_in_dollars(content):
    return f"${content}$"

# Helper function to process basis labels (e.g., e1 -> e_{1})
def process_basis_label(label):
    # Use regex to split the name and numeric suffix
    match = re.match(r'(.*?)(\d+)?$', label)
    basis_elem_name = match.group(1)  # Alphabetic part
    basis_elem_number = match.group(2)  # Numeric part, if any
    
    # Remove trailing underscores from the name
    basis_elem_name = basis_elem_name.rstrip('_')
    
    # Build the LaTeX string
    if basis_elem_number:
        return f"{convert_to_greek(basis_elem_name)}_{{{basis_elem_number}}}"
    else:
        return f"{convert_to_greek(basis_elem_name)}"