from pathlib import Path
from Workflow_DataProcessing import data_processing_workflow
from Workflow_BayesOpt import bayesopt_workflow

"""
This file should be run to execute the entire workflow (processing and upload of all completed experiments to the 
database, followed by the generation of new recommendations in the database through Bayesian Optimization). Manually
executing this file (rather than running it automatically upon completion of a new experimental observation) enables
human-in-the-loop sanity checks of the experimental workflows (i.e. identifying obvious failures in synthesis, HPLC-MS
analysis, or spectroscopic characterization). 
"""

######### SPECIFY THE PATHS TO THE RESPECTIVE FILES AND FOLDERS HERE #########

# Path to the folder containing the metadata files for the completed runs (as generated by the ChemspeedOperator)
completed_runs_dir = Path("$COMPLETED_RUNS_DIR")

# Path to the HPLC directories (raw data and column info json file)
hplc_raw_data_dir = Path("$HPLC_RAW_DATA_DIR")
hplc_column_info = Path("$HPLC_COLUMN_INFO")

# Path to the optics directories (raw data and settings json file)
optics_raw_data_dir = Path("$OPTICS_RAW_DATA_DIR")
optics_settings = Path("$OPTICS_SETTINGS")

# Path to the folder containing the json file with the BayesOpt settings
bo_settings_file = Path("$BO_SETTINGS_FILE")

##############################################################################


if __name__ == "__main__":

    # Run the data processing workflow
    data_processing_workflow(
        completed_runs_dir=completed_runs_dir,
        hplc_raw_data_dir=hplc_raw_data_dir,
        hplc_column_info=hplc_column_info,
        optics_raw_data_dir=optics_raw_data_dir,
        optics_settings=optics_settings
    )

    # Run the BayesOpt workflow
    bayesopt_workflow(
        bayesopt_config=bo_settings_file
    )

