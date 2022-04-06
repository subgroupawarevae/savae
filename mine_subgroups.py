import torch
import pandas as pd
from modules import (
    auxiliary as aux,
    beam_search_simple as beam_search_T,
)
from numpy import sqrt
import argparse
import sys
import os


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print (device)
    aux.create_folder(os.path.dirname(tconst.SUBGROUPS_PATH))
    # Read in latents, convert to tensor and remove the index column that was saved
    latents_df = pd.read_csv(tconst.CSV_PATH)
    latents_tensor = torch.Tensor(latents_df.values)[:, 1:]
    latents_tensor = latents_tensor.to(device)
    print (latents_tensor)

    # Bin the latents and then run beam search to find the best subgroups
    binned_latents_tensor = aux.bin_dataset(latents_tensor)
    binned_latents_tensor = binned_latents_tensor.to(device)
    results_as_df, quality, latents_to_optimize = beam_search_T.run(binned_latents_tensor, device, tconst.COVERAGE_COEFF, tconst.MIN_NUM_INDIVIDUALS_PER_SUBGROUP, tconst.NO_SAMPLES_CONSIDERED_IN_SGD_LOSS, tconst.MAX_DEPTH, tconst.BEAM_WIDTH)
    print(results_as_df)
    print(quality)

    results_as_df.to_csv(tconst.SUBGROUPS_PATH)



def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--config_id", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    import_command = "from configs import config_{} as tconst".format(args.config_id)
    exec (import_command)
    main()


