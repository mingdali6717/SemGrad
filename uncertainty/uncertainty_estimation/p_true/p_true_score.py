from transformers import LlamaForCausalLM, LlamaTokenizer
import math
import torch
from tqdm import tqdm
from uncertainty.utils import PromptTemplate, LLM

DEFAULT_P_TRUE_CONFIG = {
        "cached_path": None,
        "save_path": None,
        "model_name": None,
        "few_shot": True
        }

def construct_p_true_prompt(questions, responses, model_name, few_shot=True):
    prompts = []
    # prompt_template = PromptTemplate(model_name=model_name, template="{query}", use_system_message=False)
    for question, response in zip(questions, responses):
        prompt = ''
        # few_shot = "Question: Who was the third president of the United States?\nPossible answer: James Monroe\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: B\n\nQuestion: Calculate 33 + 4\nPossible answer: 37\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: A\n\nQuestion: Fill in the blank in the sentence 'I went to the grocery and then to the pharmacy. I was disappointed that they didn't have any vegetarian sausage at the _____.'\nPossible answer: grocery\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: A\n\nQuestion: Name a celebrated civil rights leader.\nPossible answer: Martin Luther King\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: A\n\nQuestion: Calculate 33 * 849\nPossible answer: 28347\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: B\n\nQuestion: Fill in the blank in the sentence 'I shot the _____ and it went swish. We walked away the winners of that battle!'\nPossible answer: gun\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: B\n\n"
        few_shot_demons = "Question: Who was the third president of the United States?\nPossible answer: James Monroe\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: False\n\nQuestion: Calculate 33 + 4\nPossible answer: 37\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: True\n\nQuestion: Fill in the blank in the sentence 'I went to the grocery and then to the pharmacy. I was disappointed that they didn't have any vegetarian sausage at the _____.'\nPossible answer: grocery\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: True\n\nQuestion: Name a celebrated civil rights leader.\nPossible answer: Martin Luther King\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: True\n\nQuestion: Calculate 33 * 849\nPossible answer: 28347\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: False\n\nQuestion: Fill in the blank in the sentence 'I shot the _____ and it went swish. We walked away the winners of that battle!'\nPossible answer: gun\nIs the possible answer:\nA) True\nB) False\nThe possible answer is: False\n\n"
        if few_shot:
            prompt += few_shot_demons
        prompt += 'Question: ' + question + '\n'
        prompt += 'Possible answer: ' + response + '\n'
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
        # prompt = prompt_template.build_prompt({"query": prompt})
        prompts.append(prompt)


    return prompts

def get_p_true(input_data, model, tokenizer):
        """Get the probability of the model anwering True for the given input."""

        # input_data += ' A'
        input_data += ' True'
        # print(f'prompt:{input_data}')
        tokenized_prompt_true = tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

def calculate_p_true(questions, responses, model_name, device_name=None, few_shot=True):
    """Calculate p_true uncertainty metric."""
    pt_model_key = LLM.initial_lm(model_name, device_name)
    pt_model, pt_tokenizer = LLM.loaded_llms[pt_model_key]
    prompts = construct_p_true_prompt(questions, responses, model_name, few_shot=few_shot)
    
    p_true_scores = []
    for prompt in tqdm(prompts, desc='Calculating p_true'):
        log_prob = get_p_true(prompt, pt_model, pt_tokenizer)
        p_true_scores.append(-log_prob)
    
    return p_true_scores

if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to('cuda')

    questions = ["What is the the capital of America?","What is the the capital of America?"]
    responses = ["Washington DC", "Beijing"]
    docs = ["Washington, D.C., formally the District of Columbia and commonly known as Washington or D.C., is the capital city and federal district of the United States.[13] The city is on the Potomac River, across from Virginia, and shares land borders with Maryland to its north and east. It was named for George Washington, the first president of the United States.[14][15] The district is named for Columbia, the female personification of the nation.The U.S. Constitution in 1789 called for the creation of a federal district under the exclusive jurisdiction of the U.S. Congress. As such, Washington, D.C., is not part of any state, and is not one itself. The Residence Act, adopted on July 16, 1790, approved the creation of the capital district along the Potomac River. The city was founded in 1791, and the 6th Congress held the first session in the unfinished Capitol Building in 1800 after the capital moved from Philadelphia. In 1801, the District of Columbia, formerly part of Maryland and Virginia and including the existing settlements of Georgetown and Alexandria, was officially recognized as the federal district; initially, the city was a separate settlement within the larger district.[16] In 1846, Congress returned the land originally ceded by Virginia, including the city of Alexandria. In 1871, it created a single municipality for the remaining portion of the district.[17] There have been several unsuccessful efforts to make the district into a state since the 1880s; a statehood bill passed the House of Representatives in 2021 but was not adopted by the U.S. Senate.", "Washington, D.C., formally the District of Columbia and commonly known as Washington or D.C., is the capital city and federal district of the United States.[13] The city is on the Potomac River, across from Virginia, and shares land borders with Maryland to its north and east. It was named for George Washington, the first president of the United States.[14][15] The district is named for Columbia, the female personification of the nation.The U.S. Constitution in 1789 called for the creation of a federal district under the exclusive jurisdiction of the U.S. Congress. As such, Washington, D.C., is not part of any state, and is not one itself. The Residence Act, adopted on July 16, 1790, approved the creation of the capital district along the Potomac River. The city was founded in 1791, and the 6th Congress held the first session in the unfinished Capitol Building in 1800 after the capital moved from Philadelphia. In 1801, the District of Columbia, formerly part of Maryland and Virginia and including the existing settlements of Georgetown and Alexandria, was officially recognized as the federal district; initially, the city was a separate settlement within the larger district.[16] In 1846, Congress returned the land originally ceded by Virginia, including the city of Alexandria. In 1871, it created a single municipality for the remaining portion of the district.[17] There have been several unsuccessful efforts to make the district into a state since the 1880s; a statehood bill passed the House of Representatives in 2021 but was not adopted by the U.S. Senate."]
    scores = calculate_p_true(questions, responses, model, tokenizer, docs, with_doc=False)
    print(f'scores:{scores}')