# Vizfold Foundations

This repository has two main components:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

---

## How do I test running OpenFold / VizFold?

| Where | How |
|-------|-----|
| **Locally** | Use **`viz_attention_demo_base.ipynb`**. Open it, set `BASE_DATA_DIR` (or the template MMCIF path in the inference command), run the setup once (e.g. `bash scripts/download_alphafold_params.sh openfold/resources`), then run the cells. Cell 2 runs OpenFold + attention extraction; cells 3–4 run the VizFold visualizations. |
| **HPC with CyberShuttle** | Use **`viz_attention_demo.ipynb`**. It uses Airavata magics to request a GPU runtime (`cybershuttle.yml`), then clones the [attention-viz-demo](https://github.com/vizfold/attention-viz-demo) repo and runs the same pipeline there. Set `BASE_DATA_DIR` to your cluster’s AlphaFold DB path (e.g. `/depot/itap/datasets/alphafold/db`). |
| **HPC without CyberShuttle** | Same workflow as local, but from the cluster: clone this repo, create the env (e.g. from `cybershuttle.yml` or OpenFold docs), download params, then run `run_pretrained_openfold.py` via your job scheduler (e.g. `sbatch`). Optionally run the `visualize_attention_*` scripts on the outputs, or run `viz_attention_demo_base.ipynb` in Jupyter on the cluster if available. |

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
