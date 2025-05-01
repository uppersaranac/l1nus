module purge
module load gcc cuda
# module load openmpi/4.1.5
module load python3_mpi

uv venv ~/source/masskit/.venv --no-managed-python

source .venv/bin/activate

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

uv pip install accelerate datasets evaluate hydra-core>=1.3.2 imageio>=2.37.0 imbalanced-learn>=0.13.0 jsonpickle>=4.0.5 levenshtein>=0.27.1 matplotlib>=3.10.1 mlflow-skinny>=2.21.2 numba>=0.61.0 numpy pandas>=2.2.3 polars>=1.26.0 pyarrow>=19.0.1 pytorch-lightning rdkit>=2024.9.6 requests>=2.32.3 rich scikit-learn>=1.6.1 scipy>=1.15.2 transformers torchmetrics tqdm>=4.67.1
