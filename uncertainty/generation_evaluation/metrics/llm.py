from ...utils import PromptTemplate  
import numpy as np
from typing import List

from openai import OpenAI
from tqdm import tqdm

DEEPSEEK_API_KEY = ""

def prompt_template_1(q, ans, c):
    """
    Question: {q}
    Correct Answers: 
    {an1}  
    {an2}
    {an3}
    Candidate Answer:{candidate}

    Is candidate correct
    """
    answers = "\n".join(ans)
    if not q.endswith("?"):
        q += "?"

    prompt = f"Question: {q}\nCorrect Answers:\n {answers}\nCandidate: {c}\n\nIs candidate answer correct?"
    return prompt

def format_evaluation_prompts(questions, answers, candidates, tempelate_index = "1"):

    """Formats prompt for fine-tuned end-to-end truth/info scores with GPT-3"""
    assert len(questions) == len(answers) and  len(questions) == len(candidates),"questions, answers and candidates should be with same size"
    prompts = []
    for q, an, c in zip(questions, answers, candidates):
        prompt = _format_evaluation_prompt(q, an, c, tempelate_index=tempelate_index)
        prompts.append(prompt)
    return prompts

def _format_evaluation_prompt(q, ans, c, tempelate_index = "1"):
    ans = [an.strip() for an in ans]
    q = q.strip()
    c = c.strip()
    
    return prompt_template_mapping["1"](q, ans, c)



prompt_template_mapping = {
    "1": prompt_template_1,
}

def llm_evalutor(questions: List[str], answers:List[List[str]], candidates:List[str]):
    """

    Uses text-davinci-003 to evaluate the correctness of candidate answers 
    

    The score is 1 if correct in model response

    questions: Column name of model answers (populate before running metrics)
    answers: ground truth answers
    candidates: generated candidate answer
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    model = "deepseek-chat"
    prompts = format_evaluation_prompts(questions, answers, candidates)
    responses = []

    prompt_template = PromptTemplate(model, template="{query}", system_message="Please judge the correctness of the candidate answer based on the several possible correct answers given, and answer directly 'yes' or 'no'.")
    for p in tqdm(prompts, desc="llm evaluator"):
        input_p = prompt_template.build_prompt({"query": p})
        chat_completion = client.chat.completions.create(
                        messages=input_p,
                        model=model,
                        temperature=1.0,
                        max_tokens = 10,
                        n=1
                    )
        responses.append(chat_completion.choices[0].message.content)

    
    is_correct = []
    for idx, r in enumerate(responses):
        if r.strip().lower().startswith("yes"):
            is_correct.append(1)
        elif r.strip().lower().startswith("no"):
            is_correct.append(0)
        else:
            print(f"Invalid response to prompt:\n\n `{prompts[idx]}`\n\nresponse:{r}")
            is_correct.append(0)

    
    return is_correct
