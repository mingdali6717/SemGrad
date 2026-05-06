from uncertainty.utils import PromptTemplate
from uncertainty.utils import LLM
import copy
from uncertainty.response_generator.base import BaseGenerator
from uncertainty.response_generator.utils import LLM_RESULTS  



qa_template = {
    1: "{query}",
    2: "Please directly answer the following question with one or few words:\n{query}", 
}

SYSTEM_MESSAGE = {0: None,
                1: "You are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me the answer. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'",
                2: "You are a highly intelligent question answering bot. If you were asked a question that is rooted in truth, you will give me a **brief, short and accurate answer with one or few words **. If you were asked a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'."}

class StandardGenerator(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        
        if config["system_id"] is not None:
            self.use_system_message = True
            system_message = SYSTEM_MESSAGE[config["system_id"]]

        else:
            self.use_system_message = False
            system_message = None

        self.config = config
        self.prompt_template = PromptTemplate(model_name=config["model_name"], 
                                              template=qa_template[config["template_id"]],
                                              system_message=system_message,
                                              use_system_message=self.use_system_message) 
        self.tokenizer = None
        
        
    def _response(self, query, knowledge=None, device="gpu0"):

        if knowledge is None:
            
            prompt = self.prompt_template.build_prompt({"query": query})
        else:
            prompt = self.prompt_template.build_prompt({"query": query, "knowledge": knowledge})

        kwargs = copy.deepcopy(self.config)
        
        
        
        outputs = LLM.lm_generate(kwargs["model_name"], [prompt], kwargs["generate_kwargs"], kwargs["tokenize_kwargs"], device_name=device, verbose=self.verbose)
        
        
        
        self.outputs = outputs

        results = LLM_RESULTS.from_dict(outputs)
        results.model_name = kwargs["model_name"]
        results.config = {"generation_config": self.config["generate_kwargs"], "tokenization_config": self.config["tokenize_kwargs"]}
        results.raw_config = self.config
        return results
    

    def _batch_response(self, query_list, knowledge_list=None, device="gpu0"):
        prompts = []
        if knowledge_list is None:
            for query in query_list:
                
                prompt = self.prompt_template.build_prompt({"query": query})
                prompts.append(prompt)

        else:
            for query,  knowledge in zip(query_list, knowledge_list):
                
                prompt = self.prompt_template.build_prompt({"query": query, "knowledge": knowledge})
                prompts.append(prompt)

        kwargs = copy.deepcopy(self.config)
        
        outputs = LLM.lm_generate(kwargs["model_name"], prompts, kwargs["generate_kwargs"], kwargs["tokenize_kwargs"], device_name=device, verbose=self.verbose)
        self.outputs = outputs
        results = LLM_RESULTS.from_dict(outputs)
        # results.model_name = kwargs["model_name"]
        # results.config = {"generation_config": self.config["generate_kwargs"], "tokenization_config": self.config["tokenize_kwargs"]}
        results.raw_config = self.config
 
        return results
    
    