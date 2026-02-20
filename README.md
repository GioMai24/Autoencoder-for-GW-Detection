# <p align='center'> Autoencoder for GW Detection </p>
LSTM autoencoder implementation to detect Gravitational Waves, based and expanding on the work of [E. A. Moreno et al.](https://arxiv.org/abs/2107.12698)

## Files Description

### `scripts` folder
The development of the optuna and training scripts listed below ran in parallel for the 'univariate' and 'multivariate' modes described in the report. Merging the two resulted unfeasible, they do however share the main structure.
- `TTVmaker.py`: 'Training, Test, and Validation' sets creation. Also greatly reduces the amount of data used from the original dataset, using it whole would have not been practical.
- `optuner_uni.py` and `optuner_multi.py`: search for best hyperparameters using optuna.
- `training_uni.py` and `training_multi.py`: continue the training of the models selected from the optuning.

### `custom` folder
- `__init__.py`: empty file to initialize the package.
- `models.py`: autoencoders models as PyTorch classes.
- `tools.py`: helper functions for the scripts.

### `model_inspector.ipynb`
Jupyter notebook to inspect best models' results: training, validation, and test losses, and reconstructions.

### `daenv.yml`
Conda environment used to run the code.
