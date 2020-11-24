import wandb
import os

if __name__ == "__main__":
    run = wandb.init(project='speedchallenge', job_type='create_dataset')

    artifact = wandb.Artifact('dataset_split', type='dataset')

    artifact.add_dir('data/train', name="train")
    artifact.add_dir('data/valid', name="valid")

    run.log_artifact(artifact)