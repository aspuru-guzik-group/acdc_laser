from pathlib import Path
import pandas as pd


def load_data(
        data_file: Path,
        synthesis_succeeded: bool = False,
        characterization_succeeded: bool = False,
        dft_desc_available: bool = False
) -> pd.DataFrame:

    observations_df: pd.DataFrame = pd.read_json(data_file)

    if not synthesis_succeeded:
        # removes all entries where gain_cross_section is nan
        observations_df = observations_df[~observations_df["gain_cross_section"].isna()]
    if not characterization_succeeded:
        # removes all entries where gain_cross_section is 0 (as a result of failed characterization)
        observations_df = observations_df[~(observations_df["gain_cross_section"] == 0.0)]
    if not dft_desc_available:
        # removes all entries where no excited state descriptors have been calculated
        observations_df = observations_df[~(observations_df["excited_state_descriptors"].str.len() == 0)]

    return observations_df
