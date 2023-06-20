import subprocess
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/total_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.000001,
            'max': 0.01
        },
        'batch_size': {
            'values': [32, 64]
        },
     #   'dropout': {
      #      'values': [0.15, 0.25, 0.5]
       # },

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
        #f'model_config.viva_model.dropout={config.dropout}',
        # add any other command line overrides you need here
    ]

    # Run the command
    subprocess.run(command)

wandb.agent(sweep_id, function=train, project="viva_sweeps", count=10)
