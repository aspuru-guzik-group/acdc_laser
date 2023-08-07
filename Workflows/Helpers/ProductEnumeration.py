import itertools
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdchiral.main import rdchiralRunText

SINGLE = Chem.rdchem.BondType.SINGLE
DOUBLE = Chem.rdchem.BondType.DOUBLE
ENDUPRIGHT = Chem.rdchem.BondDir.ENDUPRIGHT
STEREOZ = Chem.BondStereo.STEREOZ
STEREOE = Chem.BondStereo.STEREOE
SUZUKI_BOH2 = "([C,c:1]B([OH])[OH].[Br,I][C,c:2])>>[C,c:1][C,c:2]"
SUZUKI_BPIN = "([C,c:1]B1OC(C)(C)C(C)(C)O1.[Br,I][C,c:2])>>[C,c:1][C,c:2]"
SUZUKI_BMIDA = "([C,c:1]B1OC(=O)CN(C)CC(=O)O1.[Br,I][C,c:2])>>[C,c:1][C,c:2]"


def generate_makable_products(
        available_fragments: Dict[str, Dict[str, Dict[str, str]]],
        n_jobs: int = -1
) -> Dict[str, np.ndarray]:
    """
    Enumerates all possible products from the available fragments.
        1. Enumerates all possible products that are makable in each lab.
        2. Removes duplicates based on the HIDs.

    Structure of the input dictionary:
        {
            LAB_1: {
                "fragment_a": {"A001": SMILES, "A002": SMILES, ...},
                "fragment_b": {"B001": SMILES, "B002": SMILES, ...},
                "fragment_c": {"C001": SMILES, "C002": SMILES, ...}
            },
            LAB_2: {
                "fragment_a": {"A001": SMILES, "A002": SMILES, ...},
                "fragment_b": {"B001": SMILES, "B002": SMILES, ...},
                "fragment_c": {"C001": SMILES, "C002": SMILES, ...}
            },
            ...
        }

    Args:
        available_fragments: Dictionary containing the available fragments for each lab (see above).
        verbose: Whether to print progress bars.
        n_jobs: Number of jobs to use for parallelization.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the HIDs and SMILES strings of all possible products for each lab.
    """
    hid = np.zeros(0, dtype="U15")
    products = np.zeros(0, dtype="U5000")

    for lab, fragments in available_fragments.items():
        lab_hids, lab_products = generate_product_permutations(**fragments, n_jobs=n_jobs)
        hid = np.concatenate((hid, lab_hids))
        products = np.concatenate((products, lab_products))

    # Remove duplicates based on the HIDs
    _, unique_idx = np.unique(hid, return_index=True)
    hid = hid[unique_idx]
    products = products[unique_idx]

    return {"hid": hid, "smiles": products}


def generate_product_permutations(
        fragment_a: Dict[str, str],
        fragment_b: Dict[str, str],
        fragment_c: Dict[str, str],
        n_jobs: int = -1,
        verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates all possible products from the A, B and C fragments using a double Suzuki-Miyaura coupling. Product
    generation is done using the RDChiral library.

    Args:
        fragment_a: Dictionary containing the SMILES strings of the A fragments (must contain either a B(OH)2 or a
                    B(pin) group)
        fragment_b: Dictionary containing the SMILES strings of the B fragments (must contain a B(MIDA) and a halide)
        fragment_c: Dictionary containing the SMILES strings of the C fragments (must two halides)
        n_jobs: Number of jobs to use for parallelization.

    Returns:
        np.ndarray: Array containing the SMILES strings of all possible products
    """
    def generate_product(a_hid, b_hid, c_hid, a_smiles, b_smiles, c_smiles):
        hid = a_hid + b_hid + c_hid
        smiles = generate_abcba_pentamer(a_smiles, b_smiles, c_smiles)
        return [hid, smiles]

    results = np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(generate_product)(a, b, c, a_smiles, b_smiles, c_smiles)
            for (a, a_smiles), (b, b_smiles), (c, c_smiles) in tqdm(itertools.product(fragment_a.items(), fragment_b.items(), fragment_c.items()), disable=not verbose)
        )
    )
    return results[:, 0], results[:, 1]


def get_double_bond_stereochemistry(smiles: str) -> Optional[str]:
    """
    Gets the stereochemistry of the double bond in a molecule.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        str: Stereochemistry of the double bond. Either "Z" or "E". Returns None if the molecule does not contain a
             double bond or if the double bond does not have stereochemistry specified.
    """
    mol = Chem.MolFromSmiles(smiles)
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == DOUBLE and bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6]

    if len(double_bonds) == 0:
        return None
    else:
        double_bond = double_bonds[0]
        if double_bond.GetStereo() == STEREOZ:
            return "Z"
        elif double_bond.GetStereo() == STEREOE:
            return "E"
        else:
            return None


def assign_double_bond_stereochemistry(smiles: str, target_stereochemistry: Optional[str]) -> str:
    """
    Assigns a specified stereochemistry to all double bonds in a molecule.

    Args:
        smiles: SMILES string of the molecule.
        target_stereochemistry: The desired stereochemistry of the double bonds. Must be either "Z", "E" or None.

    Returns:
        str: SMILES string of the molecule with the specified stereochemistry.
    """
    if target_stereochemistry is None:
        return smiles

    mol = Chem.MolFromSmiles(smiles)

    # Get all double bonds
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == DOUBLE]

    for bond in double_bonds:
        # Get the atoms on either side of the double bond
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for adj_bond in atom1.GetBonds():
            if adj_bond.GetBondType() == SINGLE:
                adj_bond.SetBondDir(ENDUPRIGHT)  # The actual value doesn't matter as long as a direction is assigned

        for adj_bond in atom2.GetBonds():
            if adj_bond.GetBondType() == SINGLE:
                adj_bond.SetBondDir(ENDUPRIGHT)  # The actual value doesn't matter as long as a direction is assigned

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for bond in double_bonds:
        bond.SetStereo(STEREOZ if target_stereochemistry == "Z" else STEREOE)

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def generate_abcba_pentamer(fragment_a: str, fragment_b: str, fragment_c: str) -> str:
    """
    Helper function to generate an A–B–C–B–A pentamer from the A, B and C fragments using a double Suzuki-Miyaura
    coupling. Product generation is done using the RDChiral library.

    Args:
        fragment_a: SMILES string of the A fragment (must contain either a B(OH)2 or a B(pin) group)
        fragment_b: SMILES string of the B fragment (must contain a B(MIDA) and a halide)
        fragment_c: SMILES string of the C fragment (must two halides)

    Returns:
        str: SMILES string of the A–B–C–B–A pentamer

    ATTN: This function has a problem with handling double bond stereochemistry of vinyl boronic acids / halides due to
          a bug / problem in RDKit / RDChiral (loss of double bond stereochemistry when running reactions on a double
          bond atom). The current workaround is 1) determining the double bond stereochemistry of the B fragment (since
          this is the only fragment that can contain a double bond in our case), 2) running the reaction, 3) re-
          assigning the double bond stereochemistry to all double bonds in the product. This is not ideal, but it works
          for now.
    """
    double_bond_stereochemistry = get_double_bond_stereochemistry(fragment_b)

    try:
        intermediate_1 = rdchiralRunText(SUZUKI_BOH2, f"{fragment_a}.{fragment_b}")[0]
    except IndexError:
        intermediate_1 = rdchiralRunText(SUZUKI_BPIN, f"{fragment_a}.{fragment_b}")[0]

    intermediate_2 = rdchiralRunText(SUZUKI_BMIDA, f"{intermediate_1}.{fragment_c}")[0]
    product = rdchiralRunText(SUZUKI_BMIDA, f"{intermediate_1}.{intermediate_2}")[0]

    return assign_double_bond_stereochemistry(product, double_bond_stereochemistry)


