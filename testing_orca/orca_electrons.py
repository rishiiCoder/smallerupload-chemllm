import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem

import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import re

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import re

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import re

def analyze_molecule(smiles: str):
    print("=== Generating 3D structure from SMILES ===")
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    xyz_lines = [f"{atom.GetSymbol():2} {conf.GetAtomPosition(atom.GetIdx()).x: .6f} "
                 f"{conf.GetAtomPosition(atom.GetIdx()).y: .6f} "
                 f"{conf.GetAtomPosition(atom.GetIdx()).z: .6f}"
                 for atom in mol.GetAtoms()]
    xyz_string = "\n".join(xyz_lines)

    species = [
        {"name": "neutral", "charge": 0, "multiplicity": 1},
        {"name": "cation",  "charge": 1, "multiplicity": 2},  # doublet for odd electron
        {"name": "anion",   "charge": -1, "multiplicity": 2}  # doublet for odd electron
    ]

    results_energy = {}

    # Loop over all species and run ORCA
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

        print(f"Running ORCA ({sp['name']})...")
        subprocess.run(["orca", input_file], stdout=open(output_file, "w"))
        print(f"ORCA finished for {sp['name']}.")

        # Extract final single-point energy
        energy = None
        with open(output_file, "r") as f:
            for line in f:
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[-1])
                    break
        if energy is None:
            print(f"❌ Could not extract energy for {sp['name']}")
        results_energy[sp['name']] = energy

    # Extract HOMO/LUMO from neutral molecule
    def extract_homo_lumo(filename):
        with open(filename, "r") as f:
            data = f.read()

        match = re.search(r"ORBITAL ENERGIES.*?\n\*Only", data, re.DOTALL)
        if not match:
            print("❌ Could not find orbital energies in ORCA output.")
            return None, None

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

    homo, lumo = extract_homo_lumo("neutral.out")
    if homo is None or lumo is None:
        print("❌ HOMO/LUMO could not be extracted.")
        return

    gap = lumo - homo
    hartree_to_ev = 27.2114

    IP = (results_energy["cation"] - results_energy["neutral"]) * hartree_to_ev if results_energy["cation"] else None
    EA = (results_energy["neutral"] - results_energy["anion"]) * hartree_to_ev if results_energy["anion"] else None

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

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    return results


# Example usage
analyze_molecule("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\C(=C\C=N-C2=CC=C(OC)C=C2)\C)/C")

