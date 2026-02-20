## Files Description

### `scripts` folder
The development of the optuna and training scripts listed below ran in parallel for the 'univariate' and 'multivariate' modes described in the report. Merging the two resulted unfeasible, they do however share the main structure.
- `optuner_uni.py` and `optuner_multi.py`: search for best hyperparameters using optuna.
- `training_uni.py` and `training_multi.py`: continue the training of the models selected from the optuning.
- `TTVmaker.py`: 'Training, Test, and Validation' sets creation. Also greatly reduces the amount of data used from the original dataset, using it whole would have not been practical.

### `custom` folder
- `__init__.py`: empty file to initialize the package.
- `models.py`: PyTorch classes
- `tools.py`: