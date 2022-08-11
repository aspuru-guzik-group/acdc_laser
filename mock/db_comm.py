#!/usr/bin/env python



#-----------------
# DATABASE QUERIES
#-----------------

def get_fragments(client):
    frags = client.query_database(
        [
            "molecule.hid",
            'molecule.smiles',
            'molecule.molecule_id',
            'molecule_type.name',
        ],
        joins=[
            {
                "type": "molecule_type",
                "on": {
                    "column1": "molecule.molecule_type_id",
                    "column2": "molecule_type.molecule_type_id"
                }
            },
        ],
        filters={
            "type": "molecule_type.name",
            "op": "!=",
            "value": "abc_molecule"
        },
        limit=500,
        offset=None,
    )
    return frags


def get_machines(client):
    machines = client.query_database(
            [
                'machine.name',
                'lab.name',
                'machine.machine_id',
                'machine.lab_id',
            ],
            joins=[
            {
                "type": "lab",
                "on": {
                    "column1": "machine.lab_id",
                    "column2": "lab.lab_id"
                }
            },
        ],
    )
    return machines


def get_synthesis(client):
    synthesis = [
        'machine',
        'lab',
        'synthesis',
        'molecule',
        'molecule_molecule',
        'molecule.optical_properties',
    ],


    return None
