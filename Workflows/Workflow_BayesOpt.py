from pathlib import Path
import logging
import time
import json
import numpy as np

from Helpers.DatabaseWorkflows import get_all_previous_observations, get_available_fragments, remove_recommendations, upload_new_recommendations
from Helpers.ProductEnumeration import generate_makable_products

from gnn import GNN, MolFeaturizer
from chronos import DiscreteGridOptimizer
from chronos.SurrogateModels import *
from chronos.AcquisitionFunctions import *


def bayesopt_workflow(
        bayesopt_config: Path,
):
    """
    Full workflow for running the Bayesian Optimization algorithm. This is a very lengthy function...

    Individual steps:
        1. Get and Process All Experimental Observations
        2. Fetch all Available Fragments and Enumerate the Synthesizable Search Space
        3. Encode Molecules
        4. Run Bayesian Optimization
        5. Update the database

    Args:
        bayesopt_config: Path to the json file containing the configuration for the bayesian optimization run.
    """
    # 0. Setup basic properties
    config = json.load(open(bayesopt_config, "r"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    working_dir = config["working_dir"] / f"Iteration_{config['iteration_number']}_{timestamp}"
    working_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=working_dir / f"Iteration_{timestamp}.log", level=logging.DEBUG)

    # 1. Get and Process All Experimental Observations
    logging.info(f"Starting Iteration {config['iteration_number']} at {timestamp}.")
    observations, pending, not_assigned, constraints = get_all_previous_observations(config["targets"])
    np.savez_compressed(working_dir / "observations.npz", **observations)
    np.savez_compressed(working_dir / "constraints.npz", **constraints)
    np.savez_compressed(working_dir / "pending.npz", **pending)
    np.savez_compressed(working_dir / "not_assigned.npz", **not_assigned)
    logging.info(f"Fetched {len(observations['hid'])} observations from database.")
    logging.info(f"Found {len(constraints['hid'])} acquisition constraints.")
    logging.info(f"Found {len(pending['hid'])} pending observations.")
    logging.info(f"Found {len(not_assigned['hid'])} unassigned observations.\n")

    # 2. Fetch all Available Fragments and Enumerate the Synthesizable Search Space
    logging.info("Starting product space enumeration.")
    start_time = time.time()
    fragments = get_available_fragments(*config["labs"])
    product_space = generate_makable_products(fragments, n_jobs=config["n_jobs"])
    logging.info(f"Enumerated {len(product_space['hid'])} synthesizable products in {round(time.time() - start_time)} sec.")
    constraint_idx = np.isin(product_space["hid"], constraints["hid"])
    product_space["hid"] = product_space["hid"][~constraint_idx]
    product_space["smiles"] = product_space["smiles"][~constraint_idx]
    logging.info(f"{len(product_space['hid'])} products remain after removing constraints.\n")
    np.savez_compressed(working_dir / "product_space.npz", **product_space)

    # 3. Encode Molecules
    logging.info("Starting GNN encoding.")
    featurizer = MolFeaturizer(feature_config=json.load(open(config["gnn_encoder_settings"])), verbose=False)
    gnn_model = GNN(from_file=Path(config["gnn_model"]))

    for name, molecule_set in {"observations": observations, "pending": pending, "product_space": product_space}.items():
        start_time = time.time()
        graph_features = featurizer.encode_molecules(molecule_set["smiles"], batch_size=config["gnn_batch_size"])
        molecule_set["gnn_embeddings"] = gnn_model.embed(graph_features).numpy()
        logging.info(f"Encoded {len(molecule_set['hid'])} molecules ({name}) in {round(time.time() - start_time)} sec.")
        np.savez_compressed(working_dir / f"{name}.npz", **molecule_set)

    # 4. Run Bayesian Optimization
    optimizer = DiscreteGridOptimizer(
        data_dir=working_dir.parent,
        surrogate_model=eval(config["surrogate_model"]),
        surrogate_params=config["surrogate_params"],
        iteration_number=config["iteration_number"],
        acquisition_functions=[eval(acq) for acq in config["acquisition_functions"]],
        acquisition_function_params=config["acquisition_function_params"],
        logger=logging.getLogger(),
        acquisition_batch_size=config["acquisition_batch_size"],
        acquisition_MC_sample_size=config["acquisition_MC_sample_size"]
    )

    indices, acqf_values = optimizer(
        observations_features=observations["gnn_embeddings"],
        observations_targets=observations["targets"],
        objective_index=config["objective_index"],
        search_space_features=product_space["gnn_embeddings"],
        pending_data_features=pending["gnn_embeddings"],
        batch_size=config["batch_size"]
    )

    recommended_hids = product_space["hid"][indices]
    recommended_smiles = product_space["smiles"][indices]
    np.savez_compressed(working_dir / "recommended.npz", hid=recommended_hids, smiles=recommended_smiles, acqf_values=acqf_values)
    logging.info(f"Generated {len(recommended_hids)} new recommendations.\n")

    # 5. Update Database
    remove_recommendations(not_assigned["hid"])
    logging.info(f"Removed {len(not_assigned)} old recommendations.")
    upload_new_recommendations(recommended_hids, recommended_smiles)
    logging.info(f"Updated database with {len(recommended_hids)} new recommendations.")
    config["iteration_number"] += 1
    json.dump(config, open(bayesopt_config, "w"), indent=4)


if __name__ == "__main__":
    bayesopt_workflow(Path.cwd() / "Settings" / "bayesopt_config.json")