import numpy as np
import pandas as pd

data = np.load("laser_data_mar28/all_targets.npz", allow_pickle=True)

smiles = data["smiles"]
targets = data["target"]
onehot_encoding = data["onehot"]
fragment_based_descriptors = data["desc_original"]
desc_dft = data["desc_dft_extended"]
gnn_embeddings = data["gnn_embedding_100"]
ecfp6_2048 = data["ecfp6_2048"]
mordred_desc = data["desc_mordred_all"]

indices = np.arange(len(smiles))
np.random.shuffle(indices)

smiles = smiles[indices]
targets = targets[indices]
onehot_encoding = onehot_encoding[indices].astype(np.int16)
fragment_based_descriptors = fragment_based_descriptors[indices]
desc_dft = desc_dft[indices]
gnn_embeddings = gnn_embeddings[indices]
ecfp6_2048 = ecfp6_2048[indices].astype(np.int16)
mordred_desc = mordred_desc[indices]

dataset = np.concatenate((smiles, targets), axis=1)
df = pd.DataFrame(data=dataset, columns=["smiles", "Gain Cross Section (cm^2)", "Quantum Yield", "Emission Lifetime (ns)", "Spectral Gain Factor (cm^2 s)", "Emission Wavelength (nm)"])
df.to_csv("experimental_observations.csv", index=False)

df = pd.DataFrame(data=onehot_encoding, columns=[i for i in range(onehot_encoding.shape[1])])
df.to_csv("representations/onehot_encoding.csv", index=False, header=False)

df = pd.DataFrame(data=fragment_based_descriptors, columns=[i for i in range(fragment_based_descriptors.shape[1])])
df.to_csv("representations/fragment_based_descriptors.csv", index=False, header=False)

df = pd.DataFrame(data=desc_dft, columns=[i for i in range(desc_dft.shape[1])])
df.to_csv("representations/tddft_descriptors.csv", index=False, header=False)

df = pd.DataFrame(data=gnn_embeddings, columns=[i for i in range(gnn_embeddings.shape[1])])
df.to_csv("representations/gnn_embeddings.csv", index=False, header=False)

df = pd.DataFrame(data=ecfp6_2048, columns=[i for i in range(ecfp6_2048.shape[1])])
df.to_csv("representations/ecfp6_2048.csv", index=False, header=False)

df = pd.DataFrame(data=mordred_desc, columns=[i for i in range(mordred_desc.shape[1])])
df.to_csv("representations/mordred_desc.csv", index=False, header=False)

print("")