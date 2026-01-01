from logging import config
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from google.generativeai import types
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
from google import genai
from google.genai import types
#from googlesearch import search
import time
import pathlib
import pandas as pd
import psi4
import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
import requests
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import json


save_response = ""
# DEFINING THE FUNCTIONS

def load_mol(smiles: str, smarts: bool = False):
    mol = Chem.MolFromSmarts(smiles) if smarts else Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid {'SMARTS' if smarts else 'SMILES'} string: {smiles}")
    return mol

def validate_mol(smiles: str) -> bool:
  # instructions for the lLM to use
  """
  Check for if the newly generated molecular structure is valid.

  This function is designed for a molecular designer to ensure a valid molecular structure is being created.
  It takes in a SMILE string of a newly designed molecular structure made and checks for if it is a valid strucutre.

  Args:
    smiles: the newly created molecule's SMILE string

   Returns:
    If the molecule is valid or not, if not valid, resends original structure to edit to become valid.
  """
  try:
      mol = Chem.MolFromSmiles(smiles)

      if mol is None:
        raise ValueError("Molecule is invalid")

      mol = Chem.SanitizeMol(mol)

      print("Molecule is valid")
      return True

  except Exception as e:
      return False

def predict_lambda(smiles: str):
  """
  Predicts λ_max for a given SMILES string.

  It takes in a SMILE string of a newly designed molecular structure made and predicts the value of λ_max.

  Args:
    smiles: the newly created molecule's SMILE string

   Returns:
    The predicted value of λ_max.
  """
  url = "http://127.0.0.1:8000/predict_lambda"
  response = requests.post(url, json={
            "smiles": smiles
            })
  print(response.json())
  return response.json()['predicted_nm']

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

    # when geometry setting fails: give the LLM another run of the ORCA program as the geometry will fix
    MAX_RETRIES = 2  
    retry_count = 0
    orca_successful = False

    while retry_count <= MAX_RETRIES and not orca_successful:
        
        if retry_count > 0:
            print(f"\n--- Retrying ORCA (Attempt {retry_count + 1}/{MAX_RETRIES + 1}) ---")
            time.sleep(1) 
        else:
            print("Running ORCA...")

        try:
           
            subprocess.run(["orca", input_file], stdout=open(output_file, "w"), check=True, text=True) 
            print("ORCA finished successfully.")
            orca_successful = True
            
        except subprocess.CalledProcessError as e:
            
            try:
                with open(output_file, "r") as f:
                    output_content = f.read()
            except FileNotFoundError:
                print(f"Error: ORCA failed, and output file '{output_file}' was not created.")
                return None
            
            
            if "Input geometry does not match current geometry" in output_content:
                print("ORCA failed due to geometry mismatch. Retrying as this often self-corrects...")
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print("Max retries reached. ORCA process failed permanently.")
                    return None
            else:
               
                print(f"Error running ORCA: Unexpected failure. Details: {e}")
                return None
            
        except FileNotFoundError:
            print("Error: ORCA executable not found. Ensure it is installed and in your PATH.")
            return None

    # 
    if not orca_successful:
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

            column_names = ['Transition', 'Energy_eV', 'Energy_cm-1', 'Wavelength_nm', 'R_1e40*cgs', 'MX_au', 'MY_au', 'MZ_au'] 
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
   



def predict_electronproperties(smiles: str):
    """
    Predict electron acceptance and electron donation properties (HOMO, LUMO, IP, EA). 
    
    Runs ORCA single point energy calculations for neutral, cation, and anion species 
    and includes a retry mechanism for common calculation failures.

    Args:
        smiles: The newly created SMILES string.

    Returns:
        A list containing the file path of the saved JSON dictionary, 
        or a list containing an error message string if calculation fails.
    """

    print("=== Generating 3D structure from SMILES ===")
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        return [f"ERROR: RDKit failed to generate 3D structure for SMILES. Details: {e}"]

    conf = mol.GetConformer()
    xyz_lines = [f"{atom.GetSymbol():2} {conf.GetAtomPosition(atom.GetIdx()).x: .6f} "
                 f"{conf.GetAtomPosition(atom.GetIdx()).y: .6f} "
                 f"{conf.GetAtomPosition(atom.GetIdx()).z: .6f}"
                 for atom in mol.GetAtoms()]
    xyz_string = "\n".join(xyz_lines)

    species = [
        {"name": "neutral", "charge": 0, "multiplicity": 1},
        {"name": "cation", "charge": 1, "multiplicity": 2}, 
        {"name": "anion", "charge": -1, "multiplicity": 2} 
    ]
    results_energy = {}
    MAX_RETRIES = 2  
    all_runs_successful = True

    def extract_final_energy(output_content):
        match = re.search(r"FINAL SINGLE POINT ENERGY\s+(\S+)", output_content)
        return float(match.group(1)) if match else None

    def extract_homo_lumo(output_content):
        match = re.search(r"ORBITAL ENERGIES.*?\n\*Only", output_content, re.DOTALL)
        if not match: return None, None
        
        orb_data = match.group(0).splitlines()
        energies = []
        for line in orb_data:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    occ = float(parts[1])
                    e_eh = float(parts[2])
                    energies.append((occ, e_eh))
                except ValueError:
                    continue

        occupied = [e for o, e in energies if o > 0]
        virtual = [e for o, e in energies if o == 0]

        homo = max(occupied) if occupied else None
        lumo = min(virtual) if virtual else None
        return homo, lumo
    for sp in species:
        input_file = f"{sp['name']}.inp"
        output_file = f"{sp['name']}.out"
        
        orca_input = f"""! BP86 def2-SVP TightSCF

* xyz {sp['charge']} {sp['multiplicity']}
{xyz_string}
*
"""
        with open(input_file, "w") as f:
            f.write(orca_input)

        retry_count = 0
        orca_successful = False
        
        while retry_count <= MAX_RETRIES and not orca_successful:
            if retry_count > 0:
                print(f"\n--- Retrying {sp['name']} (Attempt {retry_count + 1}/{MAX_RETRIES + 1}) ---")
                time.sleep(1) 
            else:
                print(f"Running ORCA ({sp['name']})...")

            try:
                subprocess.run(["orca", input_file], stdout=open(output_file, "w"), check=True, text=True) 
                
                with open(output_file, "r") as f:
                    output_content = f.read()
                
                energy = extract_final_energy(output_content)
                
                if energy is not None:
                    results_energy[sp['name']] = energy
                    print(f"ORCA finished successfully for {sp['name']}. Energy: {energy:.6f} Eh.")
                    orca_successful = True
                else:
                    raise Exception("ORCA finished, but energy could not be extracted.")

            except (subprocess.CalledProcessError, Exception) as e:
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        output_content = f.read()
                    
                    if "Input geometry does not match current geometry" in output_content or "Error" in str(e):
                        print(f"ORCA failed for {sp['name']} (geometry/internal error).")
                        retry_count += 1
                        if retry_count > MAX_RETRIES:
                            print(f"Max retries reached for {sp['name']}. Failed permanently.")
                            all_runs_successful = False
                            break
                    else:
                        print(f"Fatal error for {sp['name']}: {e}")
                        all_runs_successful = False
                        break
                else:
                    print(f"Fatal execution error for {sp['name']}: {e}")
                    all_runs_successful = False
                    break
            
        if not orca_successful:
             break 

    if not all_runs_successful or results_energy.get("neutral") is None:
        return [f"ERROR: Quantum chemistry calculation failed after {MAX_RETRIES + 1} attempts for one or more species (Neutral/Cation/Anion). Please redesign the molecule."]

    try:
        with open("neutral.out", "r") as f:
            neutral_content = f.read()
        homo, lumo = extract_homo_lumo(neutral_content)
        if homo is None or lumo is None:
            return ["ERROR: HOMO/LUMO extraction failed from neutral ORCA output."]
    except FileNotFoundError:
        return ["ERROR: Neutral ORCA output file not found after successful run."]

    gap = lumo - homo
    hartree_to_ev = 27.2114

    IP = (results_energy["cation"] - results_energy["neutral"]) * hartree_to_ev if results_energy.get("cation") is not None else None
    EA = (results_energy["neutral"] - results_energy["anion"]) * hartree_to_ev if results_energy.get("anion") is not None else None
    
    if IP is None or EA is None:
         return ["ERROR: Cannot calculate IP/EA due to missing Cation/Anion energy, despite successful ORCA runs."]

    results = {
        "HOMO_eV": homo * hartree_to_ev,
        "LUMO_eV": lumo * hartree_to_ev,
        "GAP_eV": gap * hartree_to_ev,
        "IP_eV": IP,
        "EA_eV": EA,
        "E_neutral_Hartree": results_energy["neutral"],
        "E_cation_Hartree": results_energy["cation"],
        "E_anion_Hartree": results_energy["anion"]
    }

    output_dir = "data_csv" 
    os.makedirs(output_dir, exist_ok=True)
    output_json_file = os.path.join(output_dir, "electronprop.json")

    try:
        with open(output_json_file, "w") as f:
            json.dump(results, f, indent=4) 
        print(f"\nSuccessfully saved properties to: {output_json_file}")
    except Exception as e:
        print(f"\nERROR: Could not save JSON file. Details: {e}")
        return [f"ERROR: Final JSON save failed. Details: {e}"]

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    return [output_json_file]


# DEFINING FUNCTIONS AS TOOLS

def make_tool(name, description, params):
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=params,
                    required=list(params.keys())
                )
            )
        ]
    )

#'Checks if molecule is a valid structure and just needs standardization.',
validate_mol_tool = make_tool(
    "validate_mol",
    "Check whether a molecule SMILES string is chemically valid. "
    "Always use this before predicting lambda.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Molecule SMILES string")
    }
)

predict_lambda_tool = make_tool(
    "predict_lambda",
    "Predicts the absorption wavelength (lambda_max, nm) for a valid molecule using Chemprop. "
    "Only call this after validation passes.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }

)

predict_dft_tool = make_tool(
    "predict_dft",
    "Predicts the light absorption properpties of a SMILE string molecule through DFT Calculations and ORCA Chem Package. "
    "Only call this after validation of molecule is passed.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }
)

predict_electronproperties_tool = make_tool(
    "predict_electronproperties",
    "Predicts the electron acception and donation properties of the SMILE string molecule through PSI5 package"
    "Only call this after validation of molecule and light absoprtion properties predicted.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }

)

# DEFINING LLM FEEDBACK LOOP MECHANISMS


TARGET_MOLECULE_COUNT = 1
MAX_ITERATIONS = 11
class MoleculeDesignStateMachine:
    """Manages the state and flow of the molecule redesign process."""

    def __init__(self, target_count: int):
        self.target_count = target_count
        self.valid_molecules = {}  
        self.rejected_molecules = {}  
        self.current_smiles = None
        self.iteration = 0

    def get_progress_message(self) -> str:
        """Generate a progress message for the model."""
        return (
            f"\n{'='*60}\n"
            f"PROGRESS: {len(self.valid_molecules)}/{self.target_count} molecules found\n"
            f"Iteration: {self.iteration}\n"
            f"{'='*60}\n"
        )

    def get_feedback_message(self, function_name: str, smiles: str, result) -> str:
        """Generate feedback message based on function results."""
        if function_name == "validate_mol":
            if result:
                return (
                    f"✓ VALIDATION PASSED for: {smiles}\n"
                    f"Next step: Call predict_dft with this exact SMILES string."
                )
            else:
                return (
                    f"✗ VALIDATION FAILED for: {smiles}\n"
                    f"The molecule is chemically invalid. Generate a new, different molecule.\n"
                    f"Progress: {len(self.valid_molecules)}/{self.target_count}"
                )
            
        if function_name == "predict_dft":
            # set up file paths and variables
            count_valid_values = 0
            total_needed_values = 6
            return_message = "" 

            electric_dipole_path = 'data_csv\electric_dipole_absorption_data.csv'
            velocity_dipole_path = 'data_csv\speed_dipole_absorption_data.csv'
            cd_spectrum_path = 'data_csv\cd_spectrum_data.csv'

            electric_dip_df = pd.read_csv(electric_dipole_path)
            velocity_dip_df = pd.read_csv(velocity_dipole_path)
            cd_spec_df = pd.read_csv(cd_spectrum_path)
            
            #analyze light absorption levels

            target_range_df = electric_dip_df[
                (electric_dip_df["Wavelength_nm"] > 400) & (electric_dip_df["Wavelength_nm"] < 900)
                ]
            
            num_total_states = len(electric_dip_df)
            num_valid_states = len(target_range_df)

            if num_valid_states > 0:
                strongest_peak = target_range_df.loc[target_range_df['fosc_D2'].idxmax()]
            
                message = (
                    f"\nWavelength is optimized: {num_valid_states} of {num_total_states}"
                    f"\n States in range: ({(num_valid_states/num_total_states) * 100:.1f}%) are in the 400-900nm range. "
                    f"\nStrongest Peak: {strongest_peak['Wavelength_nm']:.1f} nm "
                )

                return_message += message
                print(message)
                count_valid_values += 1
            else:
                message = "\nWavelength is NOT optimized: No transistions found in the 400nm to 900nm range. Redo molecule design."
                return_message += message
                print(message)

        # analyze intensity and strength of absorbance

            max_fosc = electric_dip_df["fosc_D2"].max()

            if (max_fosc > 0.5) and (max_fosc < 1.5):
                message = f"\nFosc (D2) is optimized: Max fosc is {max_fosc:.3f} within the 0.5 - 1.5 range"
                return_message += message
                print(message)
                count_valid_values += 1
            else:
                message = f"\nFosc (D2) is NOT optimized: Max fosc is {max_fosc:.3f}. This is an exception, still continue to next step"
                return_message += message
                print(message)
            
            # analyze change in molecular structure 
            max_abs_R = cd_spec_df["R_1e40*cgs"].abs().max()

            if max_abs_R > 1e-6:
                message += f"\nMolecule is optically active (chiral): Max Rotatory Strength (|R|) is non-zero ({max_abs_R:.2f})"
                return_message += message
                print(message)
                count_valid_values += 1  
            else:
                return_message += "\nMolecule is NOT optically active: Max Rotatory Strength (|R|) is zero, suggesting an achiral structure. Redo molecular design "
                print("\nMolecule is not optically active: Max |R| is zero.")

           # analyizing wavelength range
            fosc_max = electric_dip_df["fosc_D2"].max()
        
            INTENSITY_THRESHOLD = 0.1 * fosc_max

            strong_transitions_df = electric_dip_df[electric_dip_df["fosc_D2"] >= INTENSITY_THRESHOLD]

            if not strong_transitions_df.empty:
                min_nm = strong_transitions_df["Wavelength_nm"].min()
                max_nm = strong_transitions_df["Wavelength_nm"].max()

                bandwidth = max_nm - min_nm
                BANDWIDTH_TARGET = 300.0

                message = (
                    f"\nSpectral Bandwidth Analysis: The range of strong absorption (>= {INTENSITY_THRESHOLD:.4F} fosc) "
                    f"\nspans from {min_nm:.1f} nm to {max_nm:.1f} nm, "
                    f"\nresulting in a Bandwidth of {bandwidth:.1f} nm"
                )

                if bandwidth >= BANDWIDTH_TARGET:
                    message += f"\nBandwidth is OPTIMIZED (>{BANDWIDTH_TARGET:.1f} nm). "
                    count_valid_values += 1
                else:
                    message += f"\nBandwidth is NOT optimized (below {BANDWIDTH_TARGET:.1f}) nm. Redo molecular Design."
                return_message += message
                print(message)
            else:
                return_message += "\nSpectral Bandwidth Analysis: No transition states found to calculate bandwidth. Redo molecule design."
                print("\nSpectral Bandwidth Analysis: No strong transition states found")
           
            peak_energy_eV = strongest_peak['Energy_eV'] 
            ENERGY_MIN = 1.38
            ENERGY_MAX = 3.10

            if (peak_energy_eV > ENERGY_MIN) and (peak_energy_eV < ENERGY_MAX):
                message = f"\nEnergy Position is OPTIMIZED: The strongest peak is at {peak_energy_eV:.2f} eV, which is within the {ENERGY_MIN} eV to {ENERGY_MAX} eV target range. "
                return_message += message
                print(message)
                # Increase the score count
                count_valid_values += 1
            else:
                message = f"\nEnergy Position is NOT optimized: Strongest peak is at {peak_energy_eV:.2f} eV, outside the {ENERGY_MIN} eV to {ENERGY_MAX} eV target. Redo molecule design. "
                return_message += message
                print(message)
            
            strongest_d2_row = electric_dip_df.loc[electric_dip_df['fosc_D2'].idxmax()]
            strongest_transition = strongest_d2_row['Transition']
            fosc_d2 = strongest_d2_row['fosc_D2']
    
    
            fosc_p2 = velocity_dip_df[velocity_dip_df['Transition'] == strongest_transition]['fosc_P2'].iloc[0]

    # Calculate percentage difference
            percent_diff = abs(fosc_d2 - fosc_p2) / ((fosc_d2 + fosc_p2) / 2) * 100
    
    # A difference under 10% is typically considered good agreement.
            if percent_diff < 10.0:
                message = f"\nGauge Agreement is GOOD: Fosc_D2 vs Fosc_P2 for strongest peak has a difference of {percent_diff:.1f}%. "
                return_message += message
                print(message)
                count_valid_values += 1
            else:
                message = f"\nGauge Agreement is POOR: Fosc_D2 vs Fosc_P2 has a large difference ({percent_diff:.1f}%). Calculation reliability is questionable but continue to next property predictor (predict_electronproperties). "
                return_message += message
                print(message)
        
    
            ratio = (float(count_valid_values) / total_needed_values) * 100
            return_message += f"\nValid and optimized amount of values: {ratio:.2f}%"
            print(return_message)

            return return_message
    
        if function_name == predict_electronproperties:
            electronprop_file = 'data_csv\electronprop.json' 
            try:
                with open(electronprop_file, "r") as f:
                    electron_results_dict = json.load(f)
            
            except (FileNotFoundError, json.JSONDecodeError) as e:
                return f"ERROR: Could not load or parse QChem results from {electronprop_file}. Details: {e}"
            
            TARGETS = {
                "HOMO_eV": {"min": -6.5, "max": -5.5, "unit": "eV"}, # Reversed min/max for clarity
                "LUMO_eV": {"min": -4.0, "max": -3.0, "unit": "eV"}, # Reversed min/max for clarity
                "GAP_eV": {"min": 1.5, "max": 2.5, "unit": "eV"},
                "IP_eV": {"min": 5.5, "max": 6.5, "unit": "eV"},
                "EA_eV": {"min": 1.5, "max": 3.5, "unit": "eV"},
            }
    
            count_valid_values = 0
            total_needed_values = len(TARGETS) # 5
            feedback_messages = []
    
    
            for key, targets in TARGETS.items():
                try:
                    value = electron_results_dict[key]
                    is_valid = (value >= targets["min"]) and (value <= targets["max"])
            
                    # Formatting the target range string
                    target_range_str = f"{targets['min']} {targets['unit']} to {targets['max']} {targets['unit']}"
            
                    if is_valid:
                        feedback_messages.append(
                            f"\n✅ {key} is **OPTIMIZED**: {value:.3f} {targets['unit']} is within the target range ({target_range_str}). "
                        )
                        count_valid_values += 1
                    else:
                        feedback_messages.append(
                            f"\n❌ {key} is **NOT optimized**: {value:.3f} {targets['unit']} is outside the target range ({target_range_str}). "
                        )
                
                except KeyError:
                    feedback_messages.append(f"\n⚠️ Warning: Could not find '{key}' in the results dictionary.")
    
        # --- Final Scoring ---

            return_message = "\n".join(feedback_messages)
    
            ratio = (float(count_valid_values) / total_needed_values) * 100
    
            return_message += "\n" + "=" * 50
            return_message += f"\n**OVERALL ELECTRONIC PROPERTY SCORE**: {count_valid_values}/{total_needed_values} checks passed ({ratio:.2f}%)"
            return_message += "\n" + "=" * 50
    
            return return_message

    def is_complete(self) -> bool:
        """Check if we've reached the target."""
        return len(self.valid_molecules) >= self.target_count

    def print_summary(self):
        """Print final summary."""
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\n✓ Accepted Molecules ({len(self.valid_molecules)}):")
        for i, (smiles, lambda_max) in enumerate(self.valid_molecules.items(), 1):
            print(f"  {i}. {smiles}")

        if self.rejected_molecules:
            print(f"\n✗ Rejected Molecules ({len(self.rejected_molecules)}):")

def run_molecule_redesign(apikey: str, pdf_path: str):
    """Main function to run molecule redesign workflow"""

    client = genai.Client(api_key=apikey)
    print("Files uploading")
    pdf_file = client.files.upload(file=pdf_path)

    modelrole = f"""
    You are computational molecular designer specialized in photochemistry, structure-property relationships, and solar material design.

        GOAL: 
        Your goal is to design an enhanced structure of the retinal molecule and rhodopsin protein complex as a biomaterial for solar energy harvesting.

        You must enhance the retinal molecule structure with structures that enhance photoswitching properties and widen range of light absorption properties in organic molecules. 
        Add simple groups that will extend light absorption towards blue light absorption (protonated azobenzene photoswitches - azonium state), green light absorption(protonated azobenzene photoswitches - azonium state), and red light (extended conjugation chain or azobenzene photoswitches). 
        Additionally, add groups to the molecule's photoswitch motif that allow for the release of an electron specifically when the molecule's structure changes (due to light absorption)
        and the ability to accept a new electron when returning to the original molecular structure post-light-absorption electron release.
        The enhanced structure must have a stronger light absorption range than silicon material.  
       Note: Make sure groups added increases light absoprtion and transition strength: must increase FOSC D2 value to be 0.5 - 1.5

        Surrounding this retinal-derivative molecule, a rhodopsin inspired protein scaffold must be designed.
        The rhodopsin inspired protein must bind tightly to the retinal-derivative molecule and when the
        retinal-derivative molecule has a structural change due to light absorption, the protein-scaffold also has a structural change due to being
        pushed around by the internal retinal-derivative molecule it contains.
        This structural change must allow the protein-scaffold to push the electron released by the retinal-derivative, acting like a piston, into the electron hole of a solar panel.
        This structural change must also open the back of the protein scaffold, to allow a new electron to bond to the retinal-derivative molecule, while the retinal-derivative returns
        to its pre-light absorption state. 
        Once the retinal-derivative returns to the original structure, the protein-scaffold must move with the molecule into its own protein-scaffold original structure as well as it is unstressed by the retinal-derivative returning to the original state.

        RESOURCES:
        ---------
        You have access to these files:
        - AzobenzeneLightReversibilityPaper.pdf: A research paper on azobenzene photoswitch molecular structures
        - Use your personal search tools.

        TOOLS:
        -----
        - The `validate_mol` tool, which checks if a proposed SMILES structure is valid -> returns True/False.
        - The 'predict_dft' tool, which predicts various density-functional theories -> prints data related to DFT properties and returns table of transistion energies
        - The 'predict_electronproperties' tool, which predicts various electron acception and electron donation properties ->  prints out electron property data and returns dictionary of properties

        TARGET VALUES CONTEXT:
        ----------------------
        - For the molecule design, ensure that the design will reach the specified target properties
        - When property predictors are called to check if these properties are met, a feedback_message will be returned to you
        - ANALYZE the feedback message: IF the feedback_message EVER STATES "Redo molecule design"  -> IMMEDIATELY REDO MOLECULE DESIGN: STEP 1 
            - If feedback message states that each PROPERTY is OPTIMIZED and GAUGE CALCULATION is UNRELIABLE, it is OKAY: CONTINUE to the next property predictor (predict_electronproperties)
            - If feedback message states that each PROPERTY is OPTIMIZED and "Fosc (D2) is NOT Optimized" -> STILL CONTINUE TO next property_predictor
        - USE feedback_message to MAKE SPECIFIC CHANGES that will OPTIMIZED the SPECIFIC PROPERTIES not currently OPTIMIZED
            
            For predict_dft:
                1) Energy (eV): >1.38 eV and <3.10 eV
                2) Wavelength (nm): >400 nm and <900 nm
                3) R 1e40*cgs: MUST be greater than 0
                4) Bandwidth (Spectral Absorbance) Gap: AT LEAST 300
            
            For predict_electronproperties:
                1) Value 1 of results dictionary (HOMO_eV): HOMO_eV > -5.5 eV and HOMO_eV < -6.5 eV
                2) Value 2 of results dictionary (LUMO_eV): LUMO_eV > -3.0 eV and LUMO_eV < -4.0 eV
                3) Value 3 of results dictionary (GAP_eV): GAP_eV > 1.5 eV and GAP_eV < 2.5
                4) Value 4 of results dictionary (IP_eV): IP_eV > 5.5 eV and IP_eV < 6.5 eV
                5) Value 5 of results dictionary (EA_eV): EA_eV > 1.5 eV and EA_eV < 3.5 eV

        Here is the STRICT WORKFLOW you MUST follow in this specific order and flow: DO NOT RETURN TO STEP 1 UNLESS SPECIFIED
        ---------------------------------------------------------------------------------------------------------------------
        1) Modify the molecule with a structure that you have researched will theoretically improve light absorption and photo switching properties
        2) FIRSTLY explain the reason why this structure improves light absorption and provide SOURCES
        3) IMMEDIATELY AFTER: call the validate_mol tool using the modified molecule as the input 
        4) WAIT for validation result
        5) IF VALID MOLECULE -> use predict_dft to predict density-functional theory properties
        6) ONLY IF INVALID MOLECULE -> RETURN to STEP 1 to alter the molecules structure to be valid
        7) RIGHT AFTER predict_dft is called -> CHECK for target properties and that they are reaching target values through the returned feedback_message
            - IF feedback_message EVER STATES for the properties "Redo Molecule design" or that "NOT OPTIMIZED" (besides for fosc (d2) AND gauge reliability) -> IMMEDIEATELY RETURN TO STEP 1
            - IF feedback_message STATES: Each property is OPTIMIZED and gauge calculation is RELIABLE -> IMMEDIATELY MOVE TO STEP 8
            - IF feedback_message STATES: Each property is OPTIMIZED and gauge calculation is UNRELIABLE -> IMMEDIATELY MOVE TO STEP 8
            - IF feedback_message STATES: Properties are NOT OPTIMIZED and gauge calculation is RELIABLE or UNRELIABLE -> RETURN TO STEP 1
            - IF Fosc (D2) is NOT OPTIMIZED -> It is OKAY -> Continue to next property predictor
        8) AFTER predict_dft is COMPLETED and TARGET VALUES ARE CHECKED, IMMEDIATELY -> call predict_electronproperties tool to predict properties related to electron transfers
        9) CHECK for target values being reached for predict_electronproperties: IF TARGER PROPERTIES REACHED: Move on to STEP 10 OR IF TARGET PROPERTIES NOT REACHED: RETURN TO STEP 1
        10) IF predict_electronproperties has been called: All properties have been predicted for one modified molecule. Give final summary of structure
        updates to molecule and stop running.

        CRITICAL RULES:
        --------------
        - Generate ONE molecule at a tinme
        - First action is ALWAYS validate_mol
        - ALWAYS validate molecule NEVER skip validate_mol
        - NEVER generate multiple molecules in one response
        - IF predict_dft OR predict_electronproperties EVER gives an error "Error: Input geometry does not match current geometry" RUN EACH METHOD *ONE* MORE TIME AGAIN AND THE METHOD WILL WORK
        - IF returned feedback_message states "Calculation reliability is questionable", it is FINE and CONTINUE to next property predictor -> predict_electronproperties
        - USE feedback_message to MAKE SPECIFIC CHANGES that will OPTIMIZED the SPECIFIC PROPERTIES not currently OPTIMIZED
        - STOP RUNNING {TARGET_MOLECULE_COUNT} valid structure is proposed and properties are predicted!
        
        After {TARGET_MOLECULE_COUNT} valid molecules have been confirmed and EACH STEP HAS BEEN COMPLETED PER MOLECULE, summarize the results as such:

        SUMMARIZED RESULTS:
    1. SMILES Molecule [number of design]:
        a. Key structural modification:
        b. Reasoning behind modification:
        c. DFT-data printed from [predict_dft]
        d. Electron property data printed from [predict_electronproperties]

    """


    mainprompt = f"""
    Analyze the structure of the following molecule: Retinal: CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\C(=C\C=O)\C)/C

    Using your knowledge, search tools, and research paper file provided "AzobenzeneLightReversibilityPaper.pdf" propose 
    {TARGET_MOLECULE_COUNT} valid, enhanced structures of Retinal that improve, optimize, and enhance its photoswitching and light absorption (increased absorption range) properties.
    Additionally, add groups to the molecule (retinal) that allow for the release of an electron due to structure changes under light and the ability to be given a new electron when returning to the original state pre-light absorption.
    
    For the molecule design:
     - Provide its SMILE string
     - Describe the key structural change(s) and why it improves the photoswitching and light absorption properties
     - Ensure electron donation abilities to potent under light absorption and when returning to structure pre-light absorption 
     - STOP RUNNING ONCE {TARGET_MOLECULE_COUNT} is completed and give summary of structural changes made and why
      
      
      Here is the STRICT WORKFLOW you MUST follow in this specific order and flow: DO NOT RETURN TO STEP 1 UNLESS SPECIFIED
        1) Modify the molecule with a structure that you have researched will theoretically improve light absorption and photo switching properties
        2) FIRSTLY explain the reason why this structure improves light absorption and provide SOURCES
        3) IMMEDIATELY AFTER: call the validate_mol tool using the modified molecule as the input 
        4) WAIT for validation result
        5) IF VALID MOLECULE -> use predict_dft to predict density-functional theory properties
        6) ONLY IF INVALID MOLECULE -> RETURN to STEP 1 to alter the molecules structure to be valid
        7) RIGHT AFTER predict_dft is called -> MAKE SURE THE RANGE OF OF WAVELENGTH IS AS LONG AS POSSIBLE
        8) AFTER predict_dft IS COMPLETED, IMMEDIATELY -> call predict_electronproperties tool to predict properties related to electron transfers
        9) IF predict_electronproperties has been called: All properties have been predicted for one modified molecule and give final summary of structure
        updates to molecule 
        10) Design the protein scaffold to surround the retinal-derivative

        CRITICAL RULES:
        - Generate ONE molecule at a tinme
        - First action is ALWAYS validate_mol
        - ALWAYS validate molecule NEVER skip validate_mol
        - NEVER generate multiple molecules in one response
        - DONT Move onto designing next molecule UNTIL ALL TOOL PROPERTY PREDICTIONS (validate_mol -> predict_dft -> predict_electronproperties) 
        are completed for the previous molecule
        - IF predict_dft OR predict_electronproperties EVER gives an error "Error: Input geometry does not match current geometry" RUN EACH METHOD *ONE* MORE TIME AGAIN AND THE METHOD WILL WORK
        - STOP RUNNING {TARGET_MOLECULE_COUNT} valid structure is proposed and properties are predicted!

    """

    # Initialize conversation history
    contents_list = [
        "Here is the research paper on Azobenzene for analysis:",
        pdf_file,
        mainprompt
    ]
    history = contents_list.copy()

    state_machine = MoleculeDesignStateMachine(TARGET_MOLECULE_COUNT)

    print(state_machine.get_progress_message())

    while state_machine.iteration < MAX_ITERATIONS:
        state_machine.iteration += 1

        try:
            # generate response
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    #tools=[validate_mol_tool, predict_lambda_tool],
                    tools=[validate_mol_tool, predict_dft_tool, predict_electronproperties_tool],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode='ANY')
                    ),
                    system_instruction=modelrole,
                    temperature=0.7
                )
            )

            # check if model called a function
            if (response.candidates and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].function_call):

                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_args = function_call.args
                smiles = function_args.get("smiles", "")

                print(f"\n[Iteration {state_machine.iteration}] Function: {function_name}")
                print(f"SMILES: {smiles}")

                # execute the function
                if function_name == "validate_mol":
                    result = validate_mol(smiles)
                    state_machine.current_smiles = smiles if result else None

                elif function_name == "predict_dft":
                    result = predict_dft(smiles)

                elif function_name == "predict_electronproperties":
                    result = predict_electronproperties(smiles)

                else:
                    print(f"Unknown function: {function_name}")
                    continue

                # add function call to history
                history.append(types.Content(
                    role="model",
                    parts=[types.Part(function_call=function_call)]
                ))

                # add function result to history
                history.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=function_name,
                        response={"result": result}
                    ))]
                ))

                # add feedback message
                feedback = state_machine.get_feedback_message(function_name, smiles, result)
                print(feedback)

                history.append(types.Content(
                    role="user",
                    parts=[types.Part(text=feedback)]
                ))

                # check if complete
                if state_machine.is_complete():
                    print("\n Target reached! Requesting final summary...")
                    break

            # text instead of function call
            else:
                text = response.text if hasattr(response, 'text') else ""
                print(f"\n[Model Response]:\n{text}\n")

                if state_machine.is_complete():
                    print("\n" + "="*60)
                    print("FINAL SUMMARY FROM MODEL")
                    print("="*60)
                    print(text)
                    break

                # otherwise, prompt to continue
                else:
                    reminder = (
                        f"Once {TARGET_MOLECULE_COUNT} is reached, stop running if not reached -> call next property prediction function: validate_mol -> predict_dft -> predict_electronproperties -> final summary and done "
                        f"Progress: {len(state_machine.valid_molecules)}/{TARGET_MOLECULE_COUNT}\n"
                        f"Next: Generate molecule #{len(state_machine.valid_molecules) + 1}"
                    )
                    history.append(types.Content(
                        role="user",
                        parts=[types.Part(text=reminder)]
                    ))


        except Exception as e:
            print(f"\n Error in iteration {state_machine.iteration}: {e}")
            import traceback
            traceback.print_exc()
            break

    # print final summary

    state_machine.print_summary()
    
    chem_response = response.text
    try:
        with open('text_results/chemllmresponse.text', "w", encoding="utf-8") as file:
            file.write(chem_response)
        
        print(f"✅ Successfully saved LLM response to: {'text_results/chemllmresponse.text'}")
    
    except Exception as e:
        print(f"❌ ERROR: Failed to write LLM response to file. Details: {e}")


    return state_machine.valid_molecules

# RUN MODEL 

pdfpath = "AzobenzeneLightReversibilityPaper.pdf"
API_KEY=""

results = run_molecule_redesign(API_KEY, pdfpath)
print("\n" + "="*60)
print("Completed Workflow")
print("=*60")
print(f"Sucessfully designed {len(results)} molecules!")





# SECOND AGENT FOR WORDED ANALYSIS OF MOLECULAR STRUCTURAL CHANGES
    # sending LLM response to a text file


# defining the API key
client = genai.Client(api_key="")

# prompt and role
summaryagentrole = f"""
    ROLE: You are clear and organized with deep knowledge on textual analysis and photochemistry. 

    CONTEXT:
    You are given a response in a text-file from a chemical-LLM that has designed an enhanced molecular structure aiming to improve solar absorption and energy harvesting. 
    You must FIND the EXACT VALID, ENHANCED Molecular structure ONLY from 'text_results\chemllmresponse.text': do NOT create your own structure
    The structure will be found and formatted in the text response in a section exactly this where the "___" contains the molecule:
    
    '[Iteration #] Function: validate_mol
        SMILES: ___
        Molecule is valid
        ✓ VALIDATION PASSED for: ___   
        Next step: Call predict_dft with this exact SMILES string.'

        
    GOAL: You are to explain the STRUCTURE of the ENHANCED and VALID molecule found in 'text_results\chemllmresponse.text', in the section formatted above withint the "___"

    CRITICAL OBJECTIVES AND STEPS:
    1) Do NOT move on UNTIL this step is COMPLETED ACCURATELY and FOLLOWS THE CRITICAL RULES: Find the EXACT, VALID, enhanced chemical structure ONLY IN 'text_results\chemllmresponse.text'
        a) If the MOLECULES in the 'text_results\chemllmresponse.text' is next to "SMILE PARSE ERROR" or "INVALID" do NOT use this one: it is NOT the valid enhanced structure -> ONLY Use the structure that is next to "VALIDATION PASSED: "
        b) If you CANNOT find the VALID structure: DO NOT create your own -> instead state that you are having trouble finding the structure and require more details on where it is in the text file
    2) Analyze the various chemical groups on the VALID, ENHANCED structure found in the text
    3) Check for if a protein structure is given in the 'text_results\chemllmresponse.text': If not state this is not found

    CRITICAL RULES:
    - DO *NOT* CREATE YOUR OWN ENHANCED MOLECULE: FIND THE VALID, ENHANCED, MOLECULE in 'text_results\chemllmresponse.text' and analyze it -> The structure will be in the format below:
    '[Iteration _] Function: validate_mol
        SMILES: ___
        Molecule is valid
        ✓ VALIDATION PASSED for: ___   
        Next step: Call predict_dft with this exact SMILES string.'
    - USE THE MOLECULE THAT WOULD BE FOUND IN THE "___" in "✓ VALIDATION PASSED for: ___"
    
    

    Provide the output as such:

    Original SMILE STRING of Molecule:
    Enhanced SMILE STRING of Molecule:
    Explanation of STRUTURAL MOTIFS OF ENHANCED MOLECULE AND THEIR PROPERTIES:
    1) 
    2)
    etc...
    [If Protein Complex Found in Response Add This Text, if not DO NOT Add this text]:
    Explanation of CHEMICAL STRUCTURES IN PROTEIN ALLOWING FOR PISTON OF ELECTRON AND ACCEPTION OF ELECTRON FOR INNER MOLECULE:
    1) 
    2)
    etc...

"""
#testrole = "You are a concise assistant."
summaryagentprompt = f"""

    Provided in the uploaded files is the 'text_results\chemllmresponse.text' which contains the
    response from the chemical-LLM.
    The response contains the enhanced molecule and properties relating to it.

    Please explain the changes made from the original molecule RETINAL: CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\C(=C\C=O)\C)/C
    compared to the EXACT,VALID, ENHANCED MOLECULE IN 'text_results\chemllmresponse.text' made to improve light absorption properties.
   
    Important things to remember given in your model role:

   CRITICAL RULES:
    - DO *NOT* CREATE YOUR OWN ENHANCED MOLECULE: FIND THE VALID, ENHANCED, MOLECULE in 'text_results\chemllmresponse.text' and analyze it -> The structure will be in the format below:
    '[Iteration _] Function: validate_mol
        SMILES: ___
        Molecule is valid
        ✓ VALIDATION PASSED for: ___   
        Next step: Call predict_dft with this exact SMILES string.'
    - USE THE MOLECULE THAT WOULD BE FOUND IN THE "___" in "✓ VALIDATION PASSED for: ___"

    """

chemllmresponse_output = 'text_results\chemllmresponse.text'
# Check if the file exists before trying to read it
if not os.path.exists(chemllmresponse_output):
    print(f"Error: File not found at {chemllmresponse_output}")
else:
    # 2. Read the content of the local file into a string variable
    try:
        with open(chemllmresponse_output, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        file_content = None


print("LLM Running...")
summaryagentresponse = client.models.generate_content(
     model="gemini-2.5-flash",
     contents=[file_content, summaryagentprompt],
     config=types.GenerateContentConfig(
         system_instruction=summaryagentrole
     )
 )

print(summaryagentresponse.text)
output_file = summaryagentresponse.text
with open ("text_results\summaryllmresponse.text", "w", encoding="utf-8") as file:
    file.write(output_file)


