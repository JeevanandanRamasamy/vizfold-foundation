#!/usr/bin/env python
# coding: utf-8

# Setting Up the Directory

# In[ ]:


import os
# Set the directory you wish put OpenFold and Vizfold-Foundation into, please change this to your desired root directory
os.environ['ROOT_DIR'] = '~/scratch'
# Set the directory that contains OpenFold data
os.environ['DATA_DIR'] = '/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data'
# Set the directory you wish put Conda
os.environ['CONDA_INSTALL_DIR'] = '~/scratch'
# Set the module that loads a base environment with Conda and Mamba
os.environ['CONDA_MODULE'] = 'miniforge'
# Set the Mamba command to run (Conda can be used as an alternative)
os.environ['MAMBA_CMD'] = 'mamba'


# Fix paths to be expanded from user to exact paths

# In[ ]:


os.environ['ROOT_DIR'] = os.path.expanduser(os.environ['ROOT_DIR'])
os.environ['DATA_DIR'] = os.path.expanduser(os.environ['DATA_DIR'])
os.environ['CONDA_INSTALL_DIR'] = os.path.expanduser(os.environ['CONDA_INSTALL_DIR'])
mamba_path = os.popen("which $MAMBA_CMD").read().strip()
os.environ['MAMBA_CMD'] = mamba_path


# Clone the Vizfold-Foundation, and OpenFold repository.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Clone the Vizfold-Foundation repository\ngit clone https://github.com/AI2Science/vizfold-foundation.git $ROOT_DIR/vizfold-foundation\n# Clone the OpenFold repository\ngit clone https://github.com/aqlaboratory/openfold.git $ROOT_DIR/openfold\n')


# Create and activate the OpenFold conda environment

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Load Miniforge module (present 3 times to flush all instances of other conda-based modules out of the module path)\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\n# Change directory to OpenFold\ncd $ROOT_DIR/openfold\n# Activate Conda\nsource "$(conda info --base)/etc/profile.d/conda.sh"\nexport CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs\nexport CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs\n$MAMBA_CMD init\n# Create the OpenFold conda environment\n$MAMBA_CMD env create -n openfold_env -f environment.yml -y\n# Activate the OpenFold conda environment\n$MAMBA_CMD activate openfold_env\n# Add pip dependencies not installed by Mamba\npip install -v deepspeed==0.14.5 dm-tree==0.1.6 git+https://github.com/NVIDIA/dllogger.git https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl --no-build-isolation\necho "openfold_env created and activated"\n')


# Set up compiler and library paths

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Load Miniforge module\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\n# Activate Conda\nsource "$(conda info --base)/etc/profile.d/conda.sh"\nexport CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs\nexport CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs\n$MAMBA_CMD init\n$MAMBA_CMD activate openfold_env\n# Change directory to OpenFold\ncd $ROOT_DIR/openfold\n# Set up compiler and library paths\nmkdir -p $CONDA_PREFIX/x86_64-conda-linux-gnu/lib\nln -s $(realpath $CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/cc1plus) $CONDA_PREFIX/bin/cc1plus\nln -s $(realpath $CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtbeginS.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtbeginS.o\nln -s $(realpath $CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtendS.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtendS.o\nln -s $(realpath $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crti.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crti.o\nln -s $(realpath $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crtn.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtn.o\n# Install gcc and libgcc-ng\n$MAMBA_CMD install -y gcc_linux-64 libgcc-ng\n# Set up environment variables\nexport GCC_LTO_PLUGIN="$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/liblto_plugin.so"\nexport CFLAGS="-O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"\nexport CXXFLAGS="$CXXFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"\nexport CFLAGS="$CFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"\nexport LDFLAGS="$LDFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"\nexport LDFLAGS="$LDFLAGS -L$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0 -L$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"\nexport CPATH="$CONDA_PREFIX/include:$CPATH"\nexport LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"\nexport LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"\n# Install OpenFold\npip install . --no-build-isolation\n# Install third-party dependencies\nscripts/install_third_party_dependencies.sh\n')


# Set up Vizfold-Foundation

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Load Miniforge module\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\n# Activate Conda\nsource "$(conda info --base)/etc/profile.d/conda.sh"\nexport CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs\nexport CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs\n$MAMBA_CMD init\n$MAMBA_CMD activate openfold_env\n# Install additional required packages\n$MAMBA_CMD install -y ipykernel\npython -m ipykernel install --user --name=openfold_env \\\n    --env PATH "$CONDA_INSTALL_DIR/.conda/envs/openfold_env/bin:/usr/local/cuda/bin:/usr/bin:/bin" \\\n    --env LD_LIBRARY_PATH "$CONDA_INSTALL_DIR/.conda/envs/openfold_env/lib:/opt/slurm/current/lib"\n# Set up Vizfold-Foundation\nmkdir -p $ROOT_DIR/vizfold-foundation/openfold\nln -s $(realpath $ROOT_DIR/openfold/openfold/data) $ROOT_DIR/vizfold-foundation/openfold/data\n# Create necessary directories and symlinks\nmkdir -p $ROOT_DIR/vizfold-foundation/openfold/resources\nln -s $(realpath $DATA_DIR/params) $ROOT_DIR/vizfold-foundation/openfold/resources/params\nwget -N --no-check-certificate -P $ROOT_DIR/vizfold-foundation/openfold/resources https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt\n')


# Install additional visualization tools

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Load Miniforge module\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\nmodule load $CONDA_MODULE\n# Activate Conda\nsource "$(conda info --base)/etc/profile.d/conda.sh"\nexport CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs\nexport CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs\n$MAMBA_CMD init\n$MAMBA_CMD activate openfold_env\n# Install matplotlib\n$MAMBA_CMD install -y conda-forge::matplotlib\n# Set strict channel priority for consistent package resolution\nconda config --set channel_priority strict \n# Install PyMOL for molecular visualization\n$MAMBA_CMD install -y -c conda-forge -c pytorch -c nvidia pymol-open-source\n# Reset channel priority\nconda config --remove-key channel_priority\n')

