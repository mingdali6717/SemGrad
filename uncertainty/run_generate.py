
import json
import math
import os
import torch
import shutil
import torch.distributed as dist
import time
from loguru import logger
from .response_generator import StandardGenerator, collect_results
from .utils import LLM, load_data, get_gpu_memory



def run_generation(rank, config, debug=False):
    # llm
    assert config["model_name"] in LLM.support_models, f"{config['model_name']} is not supported, only following models are supported:\n{';'.join(LLM.support_models)}"
    LLM.gpu_ids = config["device"]
    LLM.ddp = config["ddp"]
    verbose = config["verbose"]
    
    save_dir = os.path.dirname(config["save_path"])

    
    if rank is None:
        world_size = 1
        rank = 0 if config["device"][0] == "cpu" else config["device"][0]
    else:
        world_size = len(config["device"])
    
    if config["device"][0] == "cpu":
        device = "cpu"
    else:
        device = f"gpu{rank}"
    
    if config["ddp"]:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.barrier()
        if rank == 0 and verbose:

            logger.info(f"********************DDP connection finished for all rank********************")

    
    # load data
    logger.info(f"********************{device}: start to load dataset from {config['data_path']}********************")
    begin_time = time.time()
    dataset = load_data(config["data_path"])

    if config.get("test_num", None) is not None:
        dataset = dataset.select(list(range(config["test_num"])))
    
    # dataset = dataset.shuffle(seed=42)
    # 

    
    if config["ddp"]:
        part_num = math.ceil(len(dataset)/world_size)
        begin, end = part_num * rank, part_num * (rank + 1)

        queries = [q.strip() for q in dataset["query"][begin:end]]
        truthful_answer_list = dataset["truthful answer"][begin:end]
        if "coqa" in config["data_path"].lower():
            queries_for_calculating_metrics = [q.strip() for q in dataset["question"][begin:end]]
        else:
            queries_for_calculating_metrics = None
        dist.barrier()
        
    else:
        queries = [q.strip() for q in dataset["query"]]
        truthful_answer_list = dataset["truthful answer"]
        if "coqa" in config["data_path"].lower() or "svamp" in config["data_path"]:
            queries_for_calculating_metrics = [q.strip() for q in dataset["question"]]
        else:
            queries_for_calculating_metrics = None
    
    if rank == 0 or not config["ddp"]:

        logger.info(f"********************Dataset Loaded for all ranks, total batch size: {len(dataset)}, batch size for each rank: {len(queries)}********************")

    generator = StandardGenerator(config)
    outputs = generator.batch_response(queries, gpu=device)
    LLM.release_all()
    outputs.ground_truth = truthful_answer_list
    outputs.queries = queries
    if "coqa" in config["data_path"].lower():
        outputs.queries_for_similarity = queries_for_calculating_metrics

    if config.get("evaluation_batch_size", None) is not None:
        batch_size = config["evaluation_batch_size"] 
    else:
        batch_size = 256
    outputs.evaluate_correctness(["em"], queries=queries_for_calculating_metrics, device_name=device, batch_size=batch_size)


    if config["ddp"]:
        logger.info(f"******************** Generation Finished for {device}*******************")
        dist.barrier()
        temp_path = os.path.join(save_dir, device)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        
        outputs.save(os.path.join(temp_path, "output.json"))

        dist.barrier()
        if rank == 0: 
            logger.info(f"***********start to collect results***********")
            outputs = collect_results(save_dir)
            logger.info(f"***********results collected***********")
        dist.barrier()
        shutil.rmtree(temp_path)
    else:
        logger.info(f"******************** Generation Finished *******************")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        outputs.save(os.path.abspath(config["save_path"]))
        
        outputs.to_excel(os.path.join(save_dir, "results.xlsx"))
    
    if rank == 0:
        # 
        # 
        # evaluator = Evaluator(metrics=["em", "bem"])
        # to_evaluate = {
            # "queries": outputs["queries"],
            # "responses": outputs["responses"],
            # "ground_truth_answers": outputs["truthful_answers"]
            # }
        # evaluator.evaluate(to_evaluate, verbose=verbose, model_name=config["model_name"])
        # evaluator.result.to_excel(save_dir)
        # outputs["bem"] = evaluator.result[outputs["queries"]]["bem"].tolist()
        # outputs["em"] = evaluator.result[outputs["queries"]]["em"].tolist()
        # outputs["info"] = config
        # 
        outputs.evaluate_correctness("bem", queries=queries_for_calculating_metrics, batch_size=batch_size)
        outputs.save(os.path.abspath(config["save_path"]))
        
        outputs.to_excel(os.path.join(save_dir, "results.xlsx"))
        


