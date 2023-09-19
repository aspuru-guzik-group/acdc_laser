from typing import Tuple, List, Optional
from pathlib import Path
import itertools
from rdkit import Chem
from xyz2mol import read_xyz_file, xyz2mol
from skspatial.objects import Point, Points, Plane, Vector
from skspatial.transformation import transform_coordinates

def parse_xyz_file(file: Path) -> Chem.Mol:
	"""
	Loads an xyz file into an RDKit Mol object.
	Uses the xyz2mol package by Jan H. Jensen (https://github.com/jensengroup/xyz2mol).
	ATTENTION: If multiple molecules are provided in a single xyz file, it currently returns the first molecule only.
	Args:
		file: Path to the xyz file.
	Returns:
		Chem.Mol: RDKit Mol object of the molecule loaded from the xyz file.
	"""
	atoms, charge, xyz_coordinates = read_xyz_file(file)
	mols: List[Chem.Mol] = xyz2mol(atoms, xyz_coordinates, charge)
	return mols[0]

def get_all_atoms(mol: Chem.Mol) -> Tuple[List[str], Points]:
	"""
	Parses all atoms from an RDKit Mol object.
	Args:
		mol: RDKit Mol object.
	Returns:
		List[str]: List of all element symbols (sorted as in the Mol object).
		Points: Skspatial Points object of all atom coordinates (sorted as in the Mol object).
	"""
	element_symbols: List[str] = []
	atom_coords: List[List[float]] = []
	geometry: Chem.Conformer = mol.GetConformer()
	for i, atom in enumerate(mol.GetAtoms()):
		coordinates = geometry.GetAtomPosition(i)
		element_symbols.append(atom.GetSymbol())
		atom_coords.append([coordinates.x, coordinates.y, coordinates.z])
	return element_symbols, Points(atom_coords)

def get_sp2_atoms(mol: Chem.Mol) -> Points:
	"""
	Parses all sp2-hybridized atoms from an RDKit Mol object.
	Args:
		mol: RDKit Mol object.
	Returns:
		Points: Skspatial Points object of all atom coordinates of sp2-hybridized atoms.
	"""
	sp2_atoms: List[List[float]] = []
	geometry: Chem.Conformer = mol.GetConformer()
	for i, atom in enumerate(mol.GetAtoms()):
		if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
			coordinates = geometry.GetAtomPosition(i)
			sp2_atoms.append([coordinates.x, coordinates.y, coordinates.z])
	return Points(sp2_atoms)

def project_to_plane(points: Points) -> Tuple[Point, Vector, Vector]:
	"""
	Fits a regression plane to a cloud of points, and projects all points into that plane.
	Identifies the vector of maximum expansion within that plane.
	Args:
		points: Skspatial Points object of the point cloud.
	Returns:
		Point: Skspatial Point object of the centroid of all points projected to the plane.
		Vector: Skspatial Vector of maximum expansion within the plane (normalized).
		Vector: Skspatial Vector of the normal vector of the plane.
	"""
	plane: Plane = Plane.best_fit(points)
	points_projected: Points = Points([plane.project_point(point) for point in points])
	max_distance: float = 0
	max_expansion_vector: Optional[Vector] = None
	for p1, p2 in itertools.combinations(points_projected, 2):
		distance: float = Point(p1).distance_point(Point(p2))
		if distance > max_distance:
			max_distance = distance
			max_expansion_vector = Vector.from_points(p1, p2)
	return points_projected.centroid(), max_expansion_vector.unit(), plane.normal

def write_xyz_file(file: Path, atom_types: List[str], atom_coords: Points) -> None:
	"""
	Writes an xyz file of a molecule.
	Args:
		atom_types: List of element symbols of the atoms present in the molecule.
		atom_coords: Skspatial Points object of all atom coordinates in the molecule (same order as the atom_types).
	"""
	with open(file, "w") as xyz_file:
		xyz_file.write(f"{len(atom_types)}\n")
		xyz_file.write("\n")
		for element, coordinates in zip(atom_types, atom_coords):
			xyz_file.write(f"{element} {coordinates[0]} {coordinates[1]} {coordinates[2]}\n")
def normalize_molecule_orientation(xyz_file: Path, target_file: Path) -> None:
	"""
	Main function to normalize the orientation of long, conjugated molecules. Loads an xyz file, normalizes the
	orientation of the molecule, and saves it as an xyz file.
		1. Fits a regression plane for all sp2-hybridized atoms in the molecule.
		2. Identifies the vector of maximum geometric expansion in that plane.
		3. Rotates the molecular coordinates to a coordinate system in which the molecule centroid is at (0, 0, 0), the
		   regression plane from (1.) is the xy plane, and the expansion vector from (2.) is aligned with the x axis.
	Args:
		xyz_file: Path to the source xyz file.
		target_file: Path to the destination xyz file.
	"""
	# Parse xyz file and read in atom coordinates from RDKit Mol object
	mol: Chem.Mol = parse_xyz_file(xyz_file)
	atom_types, atom_coords = get_all_atoms(mol)
	sp2_atoms: Points = get_sp2_atoms(mol)
	# Fit regression plane and project all points into the plane
	centroid, expansion_vector, normal_vector = project_to_plane(sp2_atoms)
	# Transform coordinates to the new coordinate system
	coords_transformed: Points = Points(transform_coordinates(
		atom_coords,
		point_origin=centroid,
		vectors_basis=[expansion_vector, expansion_vector.cross(normal_vector), normal_vector]
	))
	# Save new coordinates as xyz file
	write_xyz_file(target_file, atom_types, coords_transformed)

optdir = Path.cwd()
hessdir = optdir.parent/'hess'
xyz_in = optdir/'xtbopt.xyz'
#xyz_out = optdir/'reorient.xyz'
xyz_out = hessdir/'reorient.xyz'

normalize_molecule_orientation(xyz_file = xyz_in, target_file = xyz_out)

