from Tools.ChemicalReaction import ChemicalReaction


def run_two_step_suzuki(smiles_a: str, smiles_b: str, smiles_c: str) -> str:
    first_step = ChemicalReaction(
        "[C,c:1]B([OH])[OH].[C,c:2][Br,I]>>[C,c:1][C,c:2]",
        "[C,c:1]B1OC(C)(C)C(C)(C)O1.[C,c:2][Br,I]>>[C,c:1][C,c:2]"
    )
    second_step = ChemicalReaction(
        "[C,c:1]B1OC(=O)CN(C)CC(=O)O1.[C,c:2][Br,1]>>[C,c:1][C,c:2]"
    )
    intermediate: str = first_step(smiles_a, smiles_b)
    product: str = second_step(intermediate, smiles_c)

    return product
