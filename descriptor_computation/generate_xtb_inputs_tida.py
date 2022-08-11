import os
from pathlib import Path


def main(path: Path):
    for xyz in path.rglob('conformer.xyz'):
        conformer: Path = xyz.parent
        os.chdir(conformer)
        os.system('xtb conformer.xyz --opt')
    # # generate the geometry optimization inputs
    # jobs = []
    # functionals = ['BP86']
    # for mol_id in mols:  # mol_ids
    #     dir_ = f'mols_crest_done/{mol_id}/'
    #     confs = glob.glob(dir_ + 'conf-*/')
    #
    #     for conf in confs:
    #         xyz = load_xyz(f'{conf}/conformer.xyz')
    #         # conf_id = conf.split('/')[-1].split('_')[-1].split('.')[0]
    #         #        print(conf_id)
    #         for functional in functionals:
    #             opt_input(f'{conf}/{functional}_opt.inp', xyz)
    #             jobs.append(f'{conf}')
    #
    # with open('opt_joblist.txt', 'w') as f:
    #     for job in jobs:
    #         f.write(f'{job}\n')


if __name__ == "__main__":
    tidas: Path = Path('./mols_crest_done')
    main(tidas)
