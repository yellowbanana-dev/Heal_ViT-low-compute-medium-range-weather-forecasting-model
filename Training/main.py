import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from Data_Config.config import load_config
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="HEAL-ViT Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config)

if __name__ == "__main__":
    main()
