

class BaseGenerator:
    def __init__(self, config):
        
        self.verbose = config["verbose"] if config["verbose"] is not None else True
        
        if "num_responses_per_prompt" in config["generate_kwargs"] and config["generate_kwargs"]["num_responses_per_prompt"] is not None:
            self.n = config["generate_kwargs"]["num_responses_per_prompt"]
        else:
            self.n = 1
   
    def response(self, query, knowledge=None, gpu="gpu0"):
        return self._response(query, knowledge, gpu)
    
    def batch_response(self, queries, knowledge_list=None, gpu="gpu0"):
        return self._batch_response(queries, knowledge_list, gpu)

    def save_if_set(self, new_only=False):
        pass

    def _response(self, query, language, knowledge=None, gpu="gpu0"):
        pass

    def _batch_response(self, queries,  knowledge_list=None, gpu="gpu0"):
        pass







    