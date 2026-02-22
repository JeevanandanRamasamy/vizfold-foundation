import torch
import esm

print("Downloading/Loading unmodified ESMFold model...")
model = esm.pretrained.esmfold_v1()

# Move to GPU if available and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Moving model to {device}...")
model = model.eval().to(device)

# A short test sequence (Chignolin)
sequence = "GYDPETGTWG"
print(f"Running inference on sequence: {sequence}")

with torch.no_grad():
    # Standard ESMFold API for direct PDB generation
    pdb_output = model.infer_pdb(sequence)

output_filename = "native_test_prediction.pdb"
with open(output_filename, "w") as f:
    f.write(pdb_output)

print(f"Success! Native ESMFold prediction saved to {output_filename}")