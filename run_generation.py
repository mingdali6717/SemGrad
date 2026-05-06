import argparse
import debugpy
import torch
import yaml
import dynamic_yaml
import warnings
import numpy as np
import random


import time
import os
from loguru import logger
from tqdm import tqdm
from uncertainty.run_generate import run_generation

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='a project on uncertainty estimation')
    parser.add_argument('-c', '--config', type=str, default='config/base.yaml', help='config file(yaml) path')
    parser.add_argument('-d', '--debug', action='store_true',help='use valid dataset to debug your system')
    parser.add_argument('-dp', '--data_path', type=str, help='path to the dataset')
    parser.add_argument('-o', '--output_dir', type=str, help='the directory to save the result')
    parser.add_argument('-m', '--model_name', type=str, help='the model used to generation')
    parser.add_argument('-b', '--batch_size', type=int, help='the batch size for generation')
    parser.add_argument('--do_sample', action="store_true", help='whether do sample')
    parser.add_argument('--template_id', type=int, default=2, help='the batch size for generation')
    parser.add_argument('-p', '--top_p', type=float, help='the top-p param for sampling')
    parser.add_argument('-n', '--num_of_test_examples', type=int, help='the number of examples to generate')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("-eb", "--evaluation_batch_size",type=int, default=256, help="batch size for simantic based correctness evaluation")
    

    args, _ = parser.parse_known_args()



    
    if args.debug:
        debugpy.listen(("0.0.0.0", 14328))
        print("listen ready")
        debugpy.wait_for_client()
    
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    
    
    config = dict()
    with open(args.config, 'r', encoding='utf-8') as f:
        config.update(yaml.safe_load(dynamic_yaml.dump(dynamic_yaml.load(f.read()))))
    
    if getattr(args, "data_path", None) is not None:
        config["data_path"] = args.data_path
    if getattr(args, "output_dir", None) is not None:
        config["output_dir"] = args.output_dir
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.batch_size is not None:
        config["generate_kwargs"]["batch_size"] = args.batch_size
    if args.top_p is not None:
        config["generate_kwargs"]["top_p"] = args.top_p
    if args.num_of_test_examples is not None:
        config["test_num"] = args.num_of_test_examples
    
    config["template_id"] = args.template_id
    
    config["generate_kwargs"]["do_sample"] = args.do_sample
    
    if args.evaluation_batch_size is not None:
        config["evaluation_batch_size"] = args.evaluation_batch_size

    
    config["time"] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    config["save_path"] = os.path.join(os.path.join(config["output_dir"],config["model_name"]+"-"+config["time"]), config["save_file"])
    

    # log
        
    log_name = config.get("log_name", config["time"]) + ".log"
    if not os.path.exists("log"):
        os.makedirs("log")
    logger.remove()
    if args.debug:
        level = 'DEBUG'
    else:
        level = 'INFO'
    logger.add(os.path.join("log", log_name), level=level)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)
    logger.info(f"results will be saved to {config['save_path']}")
    
    # gpu setting
    if not torch.cuda.is_available():
        gpu = False
        if config["verbose"]:
            logger.warning("[WARNING]: GPU IS NOT AVAILABLE!!!! RUN WITH CPU.")
        config["ddp"] = False
        config["device"] = ["cpu"]
        gpu_num = 0
    else:
        gpu = config["gpu"]
    
        if gpu is None:
            gpu = [i for i in range(torch.cuda.device_count())]
        elif type(gpu) is int:
            gpu = [gpu]
        elif type(gpu) is str:
            gpu = [int(i) for i in range(len(gpu.replace(" ", "").split(',')))]
        
        gpu_num = len(gpu)
        if gpu_num > 1 and getattr(config, "ddp", False):
            config["ddp"] = True
        else:
            config["ddp"] = False
        config["device"] = gpu
        logger.info(f"RUN WITH {gpu_num} GPU. GPU IDs ARE: {gpu}")
    
    # run
    if config['ddp']:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    
        world_size = len(config["device"])
        torch.multiprocessing.spawn(run_generation, args=(config, args.debug), nprocs=world_size, join=True)
        
    else:
        
        run_generation(None, config)



       
