import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer
import time
from core.utils.utils import clear_checkpoints

def main(rank, config):
    begin = time.time()
    trainer = Trainer(rank, config)
    trainer.train_loop()
    print("Time cost : ",time.time()-begin)

if __name__ == "__main__":
    config = Config("./config/acl.yaml").get_config_dict()
    clear_checkpoints(config["classifier"]["kwargs"]["checkpoint"])
    # print("config: ", config["classifier"]["kwargs"])

    if config["n_gpu"] > 1:
        pass
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    else:
        main(0, config)
