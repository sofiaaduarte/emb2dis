import warnings
import optuna
import pandas as pd
import torch as tr
from pathlib import Path
from datetime import datetime
from train_test_model import train_test_model
from src.utils import ConfigLoader

tr.multiprocessing.set_sharing_strategy('file_system')
tr.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore",
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)

TIMESTAMP = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

def objective(trial):

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Parameters to optimize
    lr_exp = trial.suggest_int('lr_exp', -6, -3)
    params = {
        'lr': 10 ** lr_exp,
        'win_len': trial.suggest_int('win_len', 7, 25),
        'filters': trial.suggest_int('filters', 50, 400, step=50),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7),
        'n_resnet': trial.suggest_int('n_resnet', 1, 4),
    }

    # Update the config with the trial parameters
    config_loader.update(params)
    config = config_loader.get_config()

    # Build a unique base_path for this trial
    net_param = (
        f"filt{config['filters']}_ker{config['kernel_size']}_resnet{config['n_resnet']}_"
        f"win{config['win_len']}_lr{config['lr']:.0e}"
    )
    trial_name = f"trial{trial.number}_{net_param}_{TIMESTAMP}"
    base_path = Path(f"results/grid_search/{trial_name}/")
    base_path.mkdir(parents=True, exist_ok=True)

    # Run training and evaluation
    train_test_model(config, base_path)

    # Read results saved by train_test_model and extract dev AUC
    results_df = pd.read_csv(base_path / 'results.csv')
    auc_score = results_df.loc[results_df['Dataset'] == 'dev', 'auc'].values[0]

    return auc_score

def main():

    name = f"search_{TIMESTAMP}"

    # Sampler configuration
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=2,  # first N trials are random
        multivariate=True     # considers hyperparameter combinations
    )

    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(objective, n_trials=3)

    # Save study results
    output_dir = Path('results/grid_search/optuna_trials/')
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f'optuna_trials_{name}.csv', index=False)

    print("\n=== Best Trial ===")
    print(f"  AUC (dev): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

if __name__ == "__main__":
    main()