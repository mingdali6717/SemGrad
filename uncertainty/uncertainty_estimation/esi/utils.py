
import jsonlines
from .aug_func import skip_one_char
from loguru import logger
def load_paraphrase(paraphrase_path, sample_num):
    
    paraphrase_mapping = dict()
    with jsonlines.open(paraphrase_path, "r") as reader:
        for line in reader:
            if "question" in line.keys():
                para_name = "question"
            else:
                para_name = "query"
            if len(line['paraphrases']) == 0:
                logger.warning(f"There is NO paraphrased prompts for '{line[para_name]}', we will use skip_char to paraphrase {sample_num} times")
                
                extended_paraphrase = skip_one_char([line[para_name].strip()], augnum=sample_num, percent=0.5, min_char_num=1)[0]
                line['paraphrases'] = extended_paraphrase
            elif len(line['paraphrases']) < sample_num:

                logger.warning(f"There is only {len(line['paraphrases'])} paraphrased prompts for '{line[para_name]}', we will repeat the parapharse to the required sample num {sample_num}")
                extended_paraphrase = line['paraphrases'] * (sample_num//len(line['paraphrases'])) + line['paraphrases'][:sample_num%len(line['paraphrases'])]
               
                line['paraphrases'] = extended_paraphrase

                
            if "question" in line.keys():
                new_paraphrase = []
                for p in line['paraphrases']:
                    new_paraphrase.append(line["query"].replace(line["question"].strip(), p.strip()))
                line['paraphrases'] = new_paraphrase
                
            
            paraphrase_mapping[line["query"].strip()] = line['paraphrases'][:sample_num]
    
    return paraphrase_mapping
