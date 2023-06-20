import subprocess
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/viva/logit_bce',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="viva_sweeps")

def train():

    # Initialize W&B run
    wandb.init()

    # Get the current run's configuration
    config = wandb.config
    print(config)

    # Construct the command
    command = [
        'mmf_run',
        'config=projects/viva/direct.yaml',
        'model=viva_model',
        'dataset=viva',
        f'optimizer.params.lr={config.learning_rate}',
        f'training.batch_size={config.batch_size}',
        # add any other command line overrides you need here
    ]

    # Run the command
    subprocess.run(command)

wandb.agent(sweep_id, function=train, project="viva_sweeps", count=10)
