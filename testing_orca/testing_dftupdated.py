import re
import pandas as pd
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import os 

def predict_dft(smiles: str):
    """
    Predicts time-dependent density functional theory (TD-DFT) spectra of a molecule.
    
    It takes a SMILES string, runs ORCA DFT calculations, and parses the resulting 
    Electric Dipole, Velocity Dipole, and CD Spectrum data into separate CSV files.

    Args:
        smiles: SMILES string of the molecule.

    Returns: 
        A list of the sections that were processed.
    """

    smiles_mol = smiles
    mol = Chem.MolFromSmiles(smiles_mol)
    mol = Chem.AddHs(mol)  
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    xyz_lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol():2} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")
    xyz_string = "\n".join(xyz_lines)

    orca_input = f"""! BP86 def2-SVP TightSCF

    %tddft
        Nroots 10
        triplets true
    end

    * xyz 0 1
    {xyz_string}
    *
    """

    input_file = "molecule.inp"
    output_file = "moleculeO.out"

    with open(input_file, "w") as f:
        f.write(orca_input)

    print("Running ORCA...")
  
    try:
       
        subprocess.run(["orca", input_file], stdout=open(output_file, "w"), check=True, text=True) 
        print("ORCA finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running ORCA: The process failed. Details: {e}")
        return None
    except FileNotFoundError:
        print("Error: ORCA executable not found. Ensure it is installed and in your PATH.")
        return None


    try:
        with open(output_file, "r") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"FATAL ERROR: ORCA completed, but the output file '{output_file}' was not created.")
        return None
 
    sections = [
      ("Electric dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
         "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"),
        ("Velocity dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS",
         "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"),
        ("CD spectrum electric dipole", "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
         "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS") # Assuming this is the next section header
    ]

    data_table_counter = 1
    

    for name, start_key, end_key in sections:
        print(f"\n=======================================================")
        print(f"Section {data_table_counter}: {name}")
        print(f"=======================================================")
        
   
        if data_table_counter == 1:
           
            column_names = ['Transition', 'Energy_eV', 'Energy_cm-1', 'Wavelength_nm', 'fosc_D2', 'D2_au2', 'DX_au', 'DY_au', 'DZ_au']
            output_csv_filename = 'data_csv\electric_dipole_absorption_data.csv'
            
        elif data_table_counter == 2:
            column_names = ['Transition', 'Energy_eV', 'Energy_cm-1', 'Wavelength_nm', 'fosc_P2', 'P2_au2', 'PX_au', 'PY_au', 'PZ_au']
            output_csv_filename = 'data_csv\speed_dipole_absorption_data.csv'
            
        elif data_table_counter == 3:

            column_names = ['Transition', 'Energy_eV', 'Energy_cm-1', 'Wavelength_nm', 'R', 'MX_au', 'MY_au', 'MZ_au'] 
            output_csv_filename = 'data_csv\cd_spectrum_data.csv'
        

        start = data.find(start_key)
        end = data.find(end_key)
        
        print(f"\n=== {name} ===") 
        
        if start != -1 and end != -1:
    
            start_of_section = start + len(start_key)
            data_table_string = data[start_of_section:end].strip()

            print(data_table_string)
            print("-" * 50)
            
            lines = data_table_string.split('\n')
            
            data_start_line = -1
            seperator_count = 0
            for i, line in enumerate(lines):
                if re.fullmatch(r'-+', line.strip()):
                    seperator_count += 1
                    if seperator_count == 2:
                        data_start_line = i + 1 
                        break

            if data_start_line != -1:
                data_lines = lines[data_start_line:]
                processed_data = []
                
                required_numeric_cols = len(column_names) - 1 
                
        
                for line in data_lines:
                    tokens = re.split(r'\s+', line.strip())
                    
                  
                    if len(tokens) >= (3 + required_numeric_cols):
                        transition = ' '.join(tokens[:3])
                        
                 
                        numeric_data = tokens[3:3 + required_numeric_cols]
                        row = [transition] + numeric_data
                        processed_data.append(row)

                df = pd.DataFrame(processed_data, columns=column_names)

            
                numeric_cols = column_names[1:]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce') 
                
       
                df.to_csv(output_csv_filename, index=False)
                
                print(f"\nData successfully parsed and saved to {output_csv_filename}")
                print(df.head()) 
            else:
                print("Could not find the start of the data table within the section.")

        else:
            print("Section not found.")
            
        data_table_counter += 1
        
    return [ 'data_csv\electric_dipole_absorption_data.csv', 'data_csv\speed_dipole_absorption_data.csv', 'data_csv\cd_spectrum_data.csv']

print(predict_dft("O"))