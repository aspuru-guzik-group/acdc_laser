import glob
from pathlib import Path

from utils import load_xyz, spe_input


def main(path: Path):
    spe_jobs = []
    for xtb in path.rglob('xtbopt.xyz'):
        conformer: Path = xtb.parent
        xyz = load_xyz(xtb)
        spe_file: Path = conformer / 'spe_input.inp'
        spe_input(spe_file, xyz)
        spe_jobs.append(spe_file)
        # tddft_input(f'{dir_}tddft_input.inp', xyz)

    # # get the job lists
    # spe_jobs = []
    # # tddft_jobs = []
    # for mol_dir in mol_dirs:
    #     spe_jobs.append(f'{mol_dir}spe_input.inp')
    #     # tddft_jobs.append(f'{mol_dir}tddft_input.inp')

    # spe_jobs[0][:-13]
    # tddft_jobs[0][:-15]

    with open('tida_spe_joblist.txt', 'w') as f:
        for job in spe_jobs:
            # f.write(f'{job[:-13]}\n')
            f.write(f'{job}\n')

    # with open('tddft_joblist.txt', 'w') as f:
    #     for job in tddft_jobs:
    #         f.write(f'{job[:-5]}\n')


if __name__ == "__main__":
    tidas: Path = Path('tidas')
    main(tidas)
