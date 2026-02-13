import os
import zipfile
import numpy as np
import uproot
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt


# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "atlas_data")

ZIP_PATH = os.path.join(
    DATA_DIR,
    "complete_set_of_ATLAS_open_data_samples_July_2016.zip"
)

EXTRACT_PATH = os.path.join(DATA_DIR, "extracted")


# Extract if not already extracted
if not os.path.exists(EXTRACT_PATH):
    print("Extracting ATLAS data...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)


# Automatically find DataMuons.root
ROOT_FILE = None
for root, dirs, files in os.walk(EXTRACT_PATH):
    if "DataMuons.root" in files:
        ROOT_FILE = os.path.join(root, "DataMuons.root")
        break

print("Found ROOT_FILE:", ROOT_FILE)

if ROOT_FILE is None:
    raise FileNotFoundError("DataMuons.root not found anywhere after extraction.")


# Open ROOT file
with uproot.open(ROOT_FILE) as file:
    print("\nKeys in file:")
    print(file.keys())

    if "mini;1" in file:
        tree = file["mini;1"]
        print("\nBranches in TTree 'mini;1':\n")

        for branch in tree.keys():
            try:
                print(f" - {branch}: {tree[branch].interpretation.typename}")
            except Exception:
                print(f" - {branch}: type unknown")
    else:
        raise KeyError("TTree 'mini;1' not found")
    

# Check lepton multiplicity
branches_to_extract = ["lep_type"]

with uproot.open(ROOT_FILE) as file:
    tree = file["mini;1"]
    extracted_data = tree.arrays(branches_to_extract, library="ak")

row_lengths = [len(row) for row in extracted_data["lep_type"]]

print("\nLepton counts:")
for i in range(1, 6):
    print(f"{i} lep:", row_lengths.count(i))

def get_indices_lc(my_list, target_value):
    return [i for i, x in enumerate(my_list) if x == target_value]

four_leps = get_indices_lc(row_lengths, 4)
print("\nIndices with 4 leptons:", four_leps[:10])


# Extract full data (4-lepton padding)
branches_to_extract = [
    'lep_pt', 'lep_n', 'lep_type', 'lep_charge', 
    'lep_eta', 'lep_phi', 'lep_E', 'lep_z0', 'lep_ptcone30', 
    'lep_etcone20', 'lep_tracksigd0pvunbiased', 'trigE', 'trigM', 
    'scaleFactor_PILEUP', 'scaleFactor_ELE', 'scaleFactor_MUON', 
    'scaleFactor_TRIGGER', 'mcWeight', 'runNumber', 'eventNumber' 
]

with uproot.open(ROOT_FILE) as file:
    tree = file["mini;1"]
    extracted_data = tree.arrays(branches_to_extract, library="ak")

num_events = len(extracted_data)
fixed_lepton_count = 4
processed_data_np = {}

for branch_name in branches_to_extract:
    ak_array = extracted_data[branch_name]
    is_jagged = ak_array.ndim > 1

    if is_jagged:
        padded = ak.fill_none(
            ak.pad_none(ak_array, fixed_lepton_count, clip=True),
            0
        )
        processed_data_np[branch_name] = ak.to_numpy(padded)
    else:
        repeated = np.repeat(
            ak.to_numpy(ak_array),
            fixed_lepton_count
        ).reshape(num_events, fixed_lepton_count)
        processed_data_np[branch_name] = repeated

print("\nlep_pt example:")
print(processed_data_np["lep_pt"][:10])


#Turn to dataframe and save as CSV
print(type(processed_data_np))

df_dict = {}

for key, value in processed_data_np.items():
    arr = ak.to_numpy(value)

    if arr.ndim == 2 and arr.shape[1] == 4:
        df_dict[f"{key}_1"] = arr[:, 0]
        df_dict[f"{key}_2"] = arr[:, 1]
        df_dict[f"{key}_3"] = arr[:, 2]
        df_dict[f"{key}_4"] = arr[:, 3]
    else:
        df_dict[key] = arr

df = pd.DataFrame(df_dict)

output_csv = os.path.join(BASE_DIR, "extracted_4lep_data.csv")
df.to_csv(output_csv, index=False)

print("Saved:", output_csv)






