"""
Currently a work in progress, but the goal is to maintain an argparser that can run the preffered models from a the cli

    Implemented Strategies are : 

        - LSTM on USD (lstm)
        - Time-Series into Images Financial Forecasting (tsff-cnn)
"""

import argparse
from models import lstm

def argparser():
    parser = argparse.ArgumentParser(description="Select a machine learning model.")
    parser.add_argument("--strat", choices=["lstm", "tsff-cnn"], help="Select a machine learning model."),
    return parser.parse_args()

def stratLoader(args):

    if args.strat == "lstm":
        lstm.round()
    if args.strat == "tsff-cnn":
        lstm.round() 
    else:
        raise NameError("model not found")
    


if __name__ == "__main__":
    args = argparser()
    stratLoader(args)