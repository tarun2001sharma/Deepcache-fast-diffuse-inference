from torch_fidelity import calculate_metrics


def evaluate_metrics(real_dir: str, fake_dir: str):
    """
    Compute FID, Inception Score, Precision, Recall between real_dir and fake_dir.
    Returns a dict with keys: fid, is_mean, is_std, precision, recall.
    """
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        verbose=False
    )
    return {
        'FID': metrics['frechet_inception_distance'],
        'IS': metrics['inception_score_mean'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall']
    }

import os, argparse, random, torch, pandas as pd
from torchvision import datasets, transforms
from sampling import load_model, sample_class_conditional, compute_macs
from metrics import evaluate_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser