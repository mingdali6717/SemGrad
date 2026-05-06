import string
from .llm import LLM
from .openai_tools import CHAT_MODEL_LIST

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

CHATGPT_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
LLAMA3_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""
MISTRAL_DEFAULT_SYSTEM_PROMPT = """You are a pirate chatbot who always responds in pirate speak!"""
GEMMA_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""
QWEN_DEFAULT_SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""

class PromptTemplate:
    def __init__(self, model_name: str, template: str, system_message: str = None,
                use_system_message: bool = True):
        """
        language: str - specific the language of instruction, support 'en' and 'zh'
        model_name: str - model_name help to determine whether to use chat_completion format for turbo/gpt-4
        or completion for normal causal LM
        template: str - template str with placeholder around with curly bracket
        system_message: str - while model_name is turbo/gpt-4, system message is need to use chat_completions
        return_fill_info: bool - return fill stage info contains template, text, template id, task name
        template_id: str - if return fill info, template id will also be contained
        task_name: str - if return info, task name will also be contained
        """
        self.template = template
        self.placeholders = self.parse_template_placeholder(template)
        
        self.use_system_message = use_system_message
        self.model_name = model_name

        if self.model_name in CHAT_MODEL_LIST:
            if (system_message is None) and use_system_message:
                self.system_message = CHATGPT_DEFAULT_SYSTEM_PROMPT
            else:
                self.system_message = system_message

        elif ("llama2" in model_name.lower() or "llama-2" in model_name.lower()) and "chat" in model_name:
            self.tokenizer = LLM.initial_tokenizer(model_name)
            if use_system_message:
                if system_message is None:
                    self.system_message = LLAMA2_DEFAULT_SYSTEM_PROMPT
                else:
                    self.system_message = system_message
            else:
                self.system_message = None
            
        elif ("llama3" in model_name.lower() or "llama-3" in model_name.lower()) and "instruct" in model_name.lower():
            if use_system_message:
                if system_message is None:
                    self.system_message = LLAMA3_DEFAULT_SYSTEM_PROMPT
                else:
                    self.system_message = system_message
            else:
                self.system_message = None
            self.tokenizer = LLM.initial_tokenizer(self.model_name)
        elif ("mistral" in model_name.lower()) and ("instruct" in model_name.lower()):
            if use_system_message:
                if system_message is None:
                    self.system_message = MISTRAL_DEFAULT_SYSTEM_PROMPT
                else:
                    self.system_message = system_message
            else:
                self.system_message = None
            self.tokenizer = LLM.initial_tokenizer(self.model_name)
        elif ("gemma" in model_name.lower()) and ("instruct" in model_name.lower() or "it" in model_name.lower()):
            if use_system_message:
                if system_message is None:
                    self.system_message = GEMMA_DEFAULT_SYSTEM_PROMPT
                else:
                    self.system_message = system_message
            else:
                self.system_message = None
            self.tokenizer = LLM.initial_tokenizer(self.model_name)
        elif ("qwen" in model_name.lower()) and ("instruct" in model_name.lower()):
            if use_system_message:
                if system_message is None:
                    self.system_message = QWEN_DEFAULT_SYSTEM_PROMPT
                else:
                    self.system_message = system_message
            else:
                self.system_message = None
            self.tokenizer = LLM.initial_tokenizer(self.model_name)
        

    

    def build_prompt(self, text):
        """
        fill in the templates
        parameters:
        text: Dict{str: str} - key is the name of placeholder in the template, value is the str to be fill

        return:
        prompt: List[str] or List[List[dict]] - list of prompts generated. If completion, return List[str],
        if chat completion, List[List[dict]]
        info: fill info
        """
        # TODO add clean function to process query and doc in text
        return self.build_model_specific_prompt(self.fill(text))

    def fill(self, text):
        """
        fill text in prompt placeholders.
        :param text: dict - key is placeholder name, value is the str to be fill in
        :return:
        instruction: prompt with placeholders filled
        """
        assert set(self.placeholders).issubset(
            set(text.keys())), f"{set(self.placeholders) - set(text.keys())} should be given"

        prompt = self.template.format(**text)

        return prompt

    def build_model_specific_prompt(self, prompt):
        if self.model_name in CHAT_MODEL_LIST:
            
            return self.build_openai_chat_message(prompt, self.system_message)
        elif ("llama2" in self.model_name.lower() or "llama-2" in self.model_name.lower()) and "chat" in self.model_name.lower():
    
            return self.build_llama_chat_template(prompt, self.tokenizer, self.system_message)
        elif ("llama3" in self.model_name.lower() or "llama-3" in self.model_name.lower()) and "instruct" in self.model_name.lower():
    
            return self.build_llama_chat_template(prompt, self.tokenizer, self.system_message)
        elif ("mistral" in self.model_name.lower()) and ("instruct" in self.model_name.lower()):
            return self.build_llama_chat_template(prompt, self.tokenizer, self.system_message)
        elif ("gemma" in self.model_name.lower()) and ("instruct" in self.model_name.lower() or "it" in self.model_name.lower()):
            return self.build_llama_chat_template(prompt, self.tokenizer, self.system_message)
        elif ("qwen" in self.model_name.lower()) and ("instruct" in self.model_name.lower()):
            return self.build_llama_chat_template(prompt, self.tokenizer, self.system_message)
        else:
            return prompt
        
    def build_openai_chat_message(self, prompt, system_message):
        """
        change input format from completion to chat
        parameters:
        prompt: str
        """
        system_message = {"role": "system", "content": self.system_message}
        query_message = {"role": "user", "content": prompt}
        return [system_message, query_message]
    

    def build_llama_chat_template(self, prompt, tokenizer, system_message):
        if system_message is not None:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                ]
        else:
            messages = [
                {"role": "user", "content": prompt},
                ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        


    @staticmethod
    def parse_template_placeholder(template):
        """
        return the name of all placeholders in the template

        Parameters:
        template: str - normally there should be placeholder(s) around with curly bracket inside.

        Return:
        placeholders: list -  list of all placeholder name in the template

        examples:
        instructions = "请用不超过50字的摘要如实的总结下述文档,摘要应该包含用于回答'{query}'这一问题的最主要的信息：{doc}\n"
        parse_template_placeholder(instructions)

        -> ['query', 'doc']
        """
        placeholders = []

        for parse_result in string.Formatter().parse(template):
            name = parse_result[1]
            if name is None:
                continue
            elif name == "":
                raise TypeError(f"no placeholder name is given in '{template}', check curly bracket")
            else:
                placeholders.append(name)

        return placeholders
