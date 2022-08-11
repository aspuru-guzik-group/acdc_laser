import glob
from utils import load_xyz, opt_input


def main(mols: list[str]):
    # generate the geometry optimization inputs
    jobs = []
    functionals = ['BP86']
    for mol_id in mols:  # mol_ids
        dir_ = f'mols_crest_done/{mol_id}/'
        confs = glob.glob(dir_ + 'conf-*/')

        for conf in confs:
            xyz = load_xyz(f'{conf}/conformer.xyz')
            # conf_id = conf.split('/')[-1].split('_')[-1].split('.')[0]
            #        print(conf_id)
            for functional in functionals:
                opt_input(f'{conf}/{functional}_opt.inp', xyz)
                jobs.append(f'{conf}')

    with open('opt_joblist.txt', 'w') as f:
        for job in jobs:
            f.write(f'{job}\n')


if __name__ == "__main__":
    new_tidas = ['B031', 'B032', 'B033', 'B034', 'B035', 'B036', 'B041', 'B043', 'B050', 'B051', 'B052', 'B053', 'B054', 'B055', 'B057']
    main(new_tidas)
