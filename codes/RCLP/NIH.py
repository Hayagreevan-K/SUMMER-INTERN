#!/usr/bin/env python3
# NIH.py - wrapper script for NIH ChestXray14 dataset using chexpert training implementation.
import os, sys, argparse
here = os.path.dirname(__file__)
sys.path.insert(0, here)
try:
    import chexpert as chp
except Exception as e:
    print('Unable to import chexpert module. Ensure chexpert.py is in same directory.', e)
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--meta-csv', type=str, required=True)
    parser.add_argument('--tasks-json', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs-per-task', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--run-name', type=str, default='rclp_nih')
    parser.add_argument('--classes', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    chp.train_rclp(args)
