from molar import Client, ClientConfig
from molar.exceptions import MolarBackendError


class MolarInterface:
    def __init__(
        self,
        user_details,
        database_name='madness_laser_test',
        server_url='https://molar.cs.toronto.edu',
    ):
        """Initializes the client to interact with the database.

        Parameters:
            user_details (dict): Dictionary of details of the registered user. {"email": $USER_EMAIL, "password": $USER_PASSWORD}

        """
        self.config = ClientConfig(server_url=server_url, database_name=database_name, **user_details)
        self.client = Client(self.config)

        self.status_possible = ('AVAILABLE', 'ACQUIRED',  'PROCESSING',  'SYNTHESIZED',  'SHIPPED',  'RECEIVED',  'DONE', 'FAILED')


    def _try_else_return_none(function):
        """Decorator to be applied to the database query functions.
        Catches MolarBackendErrors (e.g. raised when keys/uuids are not available).

        Function returns None in these cases.
        """
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except MolarBackendError:
                return None
        return wrapper


    ################################
    # FETCH DATA FROM THE DATABASE #
    ################################

    @_try_else_return_none
    def get_synthesis_details(self, identifier):
        """Fetches the details about a specific synthesis from the database.

        Parameters:
            identifier (str): Unique identifier of the specific experiment.

        Returns:
            details (pandas.DataFrame): Dataframe of details for the respective experiment.
        """

        request = self.client.query_database(
            types=[
                "synthesis",
                "molecule_molecule.molecule_id",
                "product",
                "fragment_a",
                "fragment_b",
                "fragment_c"
            ],
            joins=[
                {
                    "type": "molecule_molecule",
                    "on": {
                        "column1": "synthesis.molecule_id",
                        "column2": "molecule_molecule.molecule_id"
                    }
                },
                {
                    "type": "product",
                    "on": {
                        "column1": "product.molecule_id",
                        "column2": "molecule_molecule.molecule_id"
                    }
                },
                {
                    "type": "fragment_a",
                    "on": {
                        "column1": "fragment_a.molecule_id",
                        "column2": "molecule_molecule.fragment_a"
                    }
                },
                {
                    "type": "fragment_b",
                    "on": {
                        "column1": "fragment_b.molecule_id",
                        "column2": "molecule_molecule.fragment_b"
                    }
                },
                {
                    "type": "fragment_c",
                    "on": {
                        "column1": "fragment_c.molecule_id",
                        "column2": "molecule_molecule.fragment_c"
                    }
                }
            ],
            aliases=[
                {
                    "type": "molecule",
                    "alias": "product"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_a"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_b"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_c"
                }
            ],
            filters={
                "type": "synthesis.synthesis_id",
                "op": "==",
                "value": identifier
            }
        )

        return request[["synthesis.synthesis_id", "synthesis.status", "synthesis.molecule_id", "fragment_a.hid", "fragment_a.smiles", "fragment_b.hid", "fragment_b.smiles", "fragment_c.hid", "fragment_c.smiles"]]


    @_try_else_return_none
    def get_syntheses(self, lab, instrument, status):
        """Fetches the details about all syntheses set to a specific status (e.g. "available") for a specific lab and instrument.

        Parameters:
            lab (str): Name of the lab
            instrument (str): Name of the instrument.
            status (str): Status of the syntheses to be fetched

        Returns:
            pending_syntheses
        """
        request = self.client.query_database(
            types=[
                "synthesis",
                "molecule_molecule.molecule_id",
                "lab.name",
                "machine",
                "product",
                "fragment_a",
                "fragment_b",
                "fragment_c"
            ],
            joins=[
                {
                    "type": "molecule_molecule",
                    "on": {
                        "column1": "synthesis.molecule_id",
                        "column2": "molecule_molecule.molecule_id"
                    }
                },
                {
                    "type": "lab",
                    "on": {
                        "column1": "synthesis.lab_id",
                        "column2": "lab.lab_id"
                    }
                },
                {
                    "type": "machine",
                    "on": {
                        "column1": "synthesis.machine_id",
                        "column2": "machine.machine_id"
                    }
                },
                {
                    "type": "product",
                    "on": {
                        "column1": "product.molecule_id",
                        "column2": "molecule_molecule.molecule_id"
                    }
                },
                {
                    "type": "fragment_a",
                    "on": {
                        "column1": "fragment_a.molecule_id",
                        "column2": "molecule_molecule.fragment_a"
                    }
                },
                {
                    "type": "fragment_b",
                    "on": {
                        "column1": "fragment_b.molecule_id",
                        "column2": "molecule_molecule.fragment_b"
                    }
                },
                {
                    "type": "fragment_c",
                    "on": {
                        "column1": "fragment_c.molecule_id",
                        "column2": "molecule_molecule.fragment_c"
                    }
                }
            ],
            aliases=[
                {
                    "type": "molecule",
                    "alias": "product"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_a"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_b"
                },
                {
                    "type": "molecule",
                    "alias": "fragment_c"
                }
            ],
            filters={
                "filters": [
                    {
                        "type": "lab.name",
                        "op": "==",
                        "value": lab
                    },
                    {
                        "type": "synthesis.status",
                        "op": "==",
                        "value": status
                    },
                    {
                        "type": "machine.name",
                        "op": "==",
                        "value": instrument
                    }
                ],
                "op": "and"
            }
        )

        return request[["synthesis.synthesis_id", "synthesis.status", "synthesis.molecule_id", "lab.name", "machine.machine_id", "product.smiles", "fragment_a.hid", "fragment_a.smiles", "fragment_b.hid", "fragment_b.smiles", "fragment_c.hid", "fragment_c.smiles"]]


    @_try_else_return_none
    def get_target_molecule(self, identifier, identifier_type="hid"):
        """Queries the database for identifying a specific target molecule.
        Returns the corresponding database entry.

        Parameters:
            identifier (str): Identifier of the molecule
            identifier_type (str): Which identifier to use ("molecule_id", "hid" (default), "smiles" or "CAS")

        Returns:
            target_mol (pd.DataFrame): Dataframe of the query result
        """
        target_mol = self.client.query_database(
            types="molecule",
            filters={
                "type": f"molecule.{identifier_type}",
                "op": "==",
                "value": identifier
            }
        )

        return target_mol


    @_try_else_return_none
    def get_fragment_details(self, identifier, identifier_type="hid"):
        """Fetches the details about a specific fragments (ID, SMILES, CAS).

        Parameters:
            identifier (str): Identifier of the molecule
            identifier_type (str): Which identifier to use ("molecule_id", "hid" (default), "smiles" or "CAS")

        Returns:
            data_dict (dict): Dictionary of fragment details ("id", "smiles", "CAS")
        """
        request = self.get_target_molecule(identifier, identifier_type)

        if request is not None:
            data_dict = {"id": request.at[0, "hid"], "smiles": request.at[0, "smiles"], "CAS": request.at[0, "CAS"], "molecule_id": request.at[0, "molecule_id"]}
            return data_dict
        else:
            raise KeyError("The corresponding fragment could not be found in the database!")


    ##############################
    # WRITE DATA TO THE DATABASE #
    ##############################

    def create_target_compound(self, smiles, fragment_a, fragment_b, fragment_c, fragment_identifier="hid"):
        """Creates all entries for the synthesis of a new target compound in the database.
            - creates the entries in the molecule, molecule_molecule, and synthesis table, linked via the molecule_id
            - entries are linked to the fragments
            - sets the synthesis status to "AVAILABLE"

        Parameters:
            smiles (str): SMILES string of the target compound
            fragment_a (str): Identifier of Fragment A (e.g. human id, database id, SMILES)
            fragment_b (str): Identifier of Fragment B (e.g. human id, database id, SMILES)
            fragment_c (str): Identifier of Fragment C (e.g. human id, database id, SMILES)
            fragment_identifier (str): Type of identifier of the fragments

        Returns:
            mol (dict): Molar event source of the molecule entry generation
            mol_mol (dict): Molar event source of the molecule_molecule entry generation
            synthesis (dict): Molar event source of the synthesis entry generation
        """
        frag_a = self.get_fragment_details(fragment_a, type=fragment_identifier)
        frag_b = self.get_fragment_details(fragment_b, type=fragment_identifier)
        frag_c = self.get_fragment_details(fragment_c, type=fragment_identifier)

        mol = self.client.create_entry(
            type="molecule",
            data={
                "smiles": smiles,
                "molecule_type_id": "a06a38b6-78b6-4fcc-8a61-3fe9d0af7cea",
                "optical_properties": {},
                "commercially_available": False,
                "hid": frag_a["hid"]+frag_b["hid"]+frag_c["hid"]
            }
        )

        mol_mol = self.client.create_entry(
            type="molecule_molecule",
            data={
                "molecule_id": mol["uuid"],
                "fragment_a": frag_a["molecule_id"],
                "fragment_b": frag_b["molecule_id"],
                "fragment_c": frag_c["molecule_id"]
            }
        )

        synthesis = self.client.create_entry(
            type="synthesis",
            data={
                "molecule_id": mol["uuid"],
                "status": "AVAILABLE"
            }
        )

        return mol, mol_mol, synthesis


    def update_synthesis_status(self, identifier, status, identifier_type="hid"):
        """Updates the status of a synthesis run.

        Parameters:
            identifier (str): Identifier of the molecule
            status (str): New status to be uploaded. P
            identifier_type (str): Which identifier to use ("molecule_id", "hid" (default), "smiles")

        """
        if status not in self.status_possible:
            raise KeyError("The desired target status is not available in the database.")

        target_molecule_id = self.get_target_molecule(identifier, identifier_type).at[0, "molecule_id"]

        target_synthesis = self.client.query_database(
            types="synthesis",
            filters={
                "type": "synthesis.molecule_id",
                "op": "==",
                "value": target_molecule_id


            }
        )

        self.client.update_entry(
            uuid=target_synthesis.at[0, "synthesis_id"],
            type="synthesis",
            data={"status": status}
        )


    def update_optics_data(self, identifier, optics_data, identifier_type="hid"):
        """Uploads the optical characterization data after completion of a synthesis run.
            - Checks for the uuid of the target molecule in the molecule table (molecule.molecule_id).
            - Writes the characterization data to the entry in the molecule table (molecule.optical_properties).
            - Updates the Synthesis status to "DONE"

        Parameters:
            identifier (str): Identifier of the synthesis run.
            optics_data (dict): Optical properties to be uploaded
            identifier_type (str): Which identifier to use ("molecule_id", "hid" (default), "smiles")
        """
        target_mol = self.get_target_molecule(identifier, identifier_type)

        if target_mol is not None:
            target_uuid = target_mol.at[0, "molecule_id"]
        else:
            raise KeyError(f"The molecule {identifier} was not found in the database.")

        self.client.update_entry(
            uuid=target_uuid,
            type="molecule",
            data={
                "optical_properties": optics_data,
            }
        )

        self.update_synthesis_status(identifier, "DONE", identifier_type)
