def load_xyz(filename):
    with open(filename, 'r') as content:
        xyz = content.readlines()[2:]
        # xyz = [_.strip('\n') for _ in xyz]
    return xyz


def opt_input(filename, xyz, **kwargs):
    charge = 0
    mult = 1

    s = f'$molecule\n{charge} {mult}\n'
    for line in xyz:
        s += f' {line}'
    s += '$end\n\n'
    s += '$rem\n'
    s += f' BASIS = 6-31G*\n'
    s += f' GUI = 2\n'
    s += f' JOBTYPE = Optimization\n'
    s += f' METHOD = B3LYP\n'
    s += f' MOLDEN_FORMAT false\n'
    s += f'$end'

    with open(filename, 'w') as f:
        f.write(s)
    f.close()


def spe_input(filename, xyz, **kwargs):
    charge, mult = 0, 1
    s = f'$molecule\n{charge} {mult}\n'
    for line in xyz:
        s += f' {line}'
    s += '$end\n\n'
    s += '$rem\n'
    s += f' BASIS = 6-31G*\n'
    s += f' GUI = 2\n'
    s += f' METHOD = B3LYP\n'
    s += f' PRINT_ORBITALS false\n'
    s += f' MOLDEN_FORMAT false\n'
    s += f'$end'

    with open(filename, 'w') as f:
        f.write(s)
    f.close()


def tddft_input(filename, xyz, **kwargs):
    charge, mult = 0, 1
    s = f'$molecule\n{charge} {mult}\n'
    for line in xyz:
        s += f' {line}'
    s += '$end\n\n'
    s += '$rem\n'
    s += f' BASIS = 6-31G*\n'
    s += f' GUI = 2\n'
    s += f' METHOD = B3LYP\n'
    s += f' PRINT_ORBITALS false\n'
    s += f' MOLDEN_FORMAT false\n'
    s += f'$end'

    with open(filename, 'w') as f:
        f.write(s)
    f.close()

# def generate_input_qchem(filename, xyz, charge, mult, omega,
#                          functional = 'LRC-wPBE', basis_set = 'def2-SVP',
#                          **kwargs):
#     omega_fmt = int(omega*1e3)
#     s = f'$molecule\n{charge} {mult}\n'
#     for line in xyz:
#         s += f' {line}'
#     s += '$end\n\n'
#     s += '$rem\n'
#     s += f' EXCHANGE  {functional}\n'
#     s += f' BASIS  {basis_set}\n'
#     s += f' LRC_DFT TRUE\n'
#     s += f' OMEGA {omega_fmt}\n'
#     s += f' PRINT_ORBITALS false\n'
#     s += f' MOLDEN_FORMAT false\n'
#     s += f'$end'

#     with open(filename, 'w') as f:
#         f.write(s)
#     f.close()