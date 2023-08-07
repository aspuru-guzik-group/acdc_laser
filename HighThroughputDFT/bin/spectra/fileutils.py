import re
import numpy as np

def geom2xyz(atoms, x, y, z, comment=""):
    """Transform a molecular geometry to a .xyz format string."""
    geomstring = "%i\n%s\n"%(len(atoms), comment)
    for i in range(len(atoms)):
        geomstring += " {atom}\t{x}\t{y}\t{z}\n".format(atom=atoms[i], x=x[i], y=y[i], z=z[i])
    geomstring = geomstring[:-1]
    return geomstring

def geom2qchem(atoms, x, y, z, charge=0, mult=1):
    """Transform a molecular geometry to qchem $mol group string."""
    geomstring = "$molecule\n %i %i\n" %(charge,mult)
    for i in range(len(atoms)):
        geomstring += " {atom}\t{x}\t{y}\t{z}\n".format(atom=atoms[i],
                                                        x=x[i],
                                                        y=y[i],
                                                        z=z[i])
    geomstring = geomstring + "$end"
    return geomstring


def qchem_parse_gradient(natom, fname):
    """Parser for gradients from a QChem force calculation.

    Parameters
    ----------
    natom : int
        Number of atoms.

    fname : str
        Filename to parse.

    Returns
    -------
    output : ndarray
        natom x 3 array of gradients. The gradient is given in
        cartesian coordinate, in units of Hartree / bohr.
    """
    
    # Find beginning of gradient section
    f = open(fname)
    loading = False
    out = []

    for line in f:
        if "Gradient of" in line:
            loading=True
            this_out = np.zeros((natom, 3))
            iatom = 0
            jcoord = 0

        # Parse out the numbers
        if loading:
            if "Gradient time" in line:
                loading = False
                out += [this_out.copy()]
            else:
                m = re.findall('(-?[0-9]*\.[0-9]*)', line)
                if len(m) > 0:
                    vals = np.array([float(s) for s in m])
                    this_out[iatom:iatom+6, jcoord] = vals[:]
                    jcoord += 1
                    if jcoord == 3:
                        jcoord = 0
                        iatom = iatom + 6

    return out 

def qchem_parse_transdip(fname):
    """Parser for transition dipoles from a tddft QChem calculation.

    Parameters
    ----------
    fname : str
        Filename to parse.

    Returns
    -------
    output : ndarray
        A dictionary of transition moments between states.
    """
    f = open(fname)
    out = {}
    for line in f:
        if "STATE-TO-STATE TRANSITION MOMENTS" in line:
            break
        
    for line in f:
        if "END OF TRANSITION MOMENT" in line:
            break

        else:
            state_ids = re.findall('\s(\d+)\s', line)
            if len(state_ids) == 2:
                mu_xyz = re.findall('(-?\d+\.\d*)', line)
                i,j = list(map(int, state_ids))
                mus = list(map(float, mu_xyz[:3]))
                # TODO CHECK IF i,j ALREADY in OUT?
                out[i,j] = np.array(mus)
                out[j,i] = np.array(mus)
                
            
    return out
