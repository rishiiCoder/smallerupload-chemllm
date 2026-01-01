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
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import io
from io import StringIO

def predict_dft(smiles: str):
    """
        Predicts time-dependent density functional theory of a molecule.

        It takes in a SMILE String and converts it to its molecular coordinates and geometry.

        Then it uses the ORCA Chem Package to run DFT Calculations on the molecule

        Args:
            smiles: newly created SMILES string

        Returns: DFT-Calculations
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
        Nroots   10
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
    subprocess.run(["orca", input_file], stdout=open(output_file, "w"))
    print("ORCA finished.")

    with open(output_file, "r") as f:
        data = f.read()

    sections = [
     ("Electric dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
                                    "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"),
        ("Velocity dipole absorption", "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS",
                                    "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"),
        ("CD spectrum electric dipole", "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
                                    "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
    ]

    output_text = "testing_orca\lightdata_text.txt"
    data_table_counter = 1
    for name, start_key, end_key in sections:

        data_string = ""
        start = data.find(start_key)

        end = data.find(end_key) if end_key else None
        print(f"\n=== {name} ===")

        if start != -1:
            print(data[start:end])
        else:
            print("Section not found.")

        if start != -1:
            string_data = data[start:end]
        else:
            print("Section not found.")

        with open(output_text, "w") as text_file:
            text_file.write(string_data)
        data_table_counter += 1
        if data_table_counter == 4:
            column_names = ['Transition', 'Energy_eV', 'Energy_cm-1', 'Wavelength_nm', 'R', 'MX_au', 'MY_au', 'MZ_au']
            lines = output_text.strip().split('\n')
            start_index = -1
            end_index = -1
            seperator_count = 0
            for i, line in enumerate(lines):
                if re.fullmatch(r'-+', line.strip()):
                    seperator_count += 1
                    if seperator_count == 2 and start_index == -1:
                        start_index = i + 1

                    elif seperator_count == 3:
                        end_index = i
                        break
            data_lines = lines[start_index:end_index]
            processed_data = []
            for line in data_lines:
                tokens = re.split(r'\s+', line.strip())

                if len(tokens) >= 10:
                    transition = ' '.join(tokens[:3])
                    numeric_data = tokens[3:]
                    row = [transition] + numeric_data
                    processed_data.append(row)

            df = pd.DataFrame(processed_data, columns=column_names)

            numeric_cols = column_names[1:]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.to_csv('cd_spectrum_data.csv', index=False)

    return sections

print(predict_dft("O"))