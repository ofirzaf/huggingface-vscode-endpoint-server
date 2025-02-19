import time
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from transformers import Pipeline, pipeline

import torch

import openvino_genai as ov_genai

from llm_pipeline_with_hf_tokenizer import LLMPipelineWithHFTokenizer


logger = logging.getLogger(__name__)


class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)

class OpenVinoGenerator(GeneratorBase):
    def __init__(self, pretrained: str, draft: str = None, device: str = 'GPU', stop_strings: list = None):
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.num_kv_blocks = 2048 // 16
        scheduler_config.dynamic_split_fuse = False
        scheduler_config.max_num_batched_tokens = 2048
        self.stop_strings = set(stop_strings) if stop_strings is not None else None
        self.num_assistant_tokens = 0
        if draft is not None:
            draft_scheduler_config = ov_genai.SchedulerConfig()
            draft_scheduler_config.num_kv_blocks = 2048 // 16
            draft_scheduler_config.dynamic_split_fuse = False
            draft_scheduler_config.max_num_batched_tokens = 2048
            self.num_assistant_tokens = 3
            draft_model = ov_genai.draft_model = ov_genai.draft_model(draft, device, scheduler_config=draft_scheduler_config)
            self.pipe = LLMPipelineWithHFTokenizer(pretrained, device, scheduler_config=scheduler_config, draft_model=draft_model)
        else:
            self.pipe = LLMPipelineWithHFTokenizer(pretrained, device, scheduler_config=scheduler_config)
        # warmup
        self.pipe.generate(['hello'], ov_genai.GenerationConfig(), max_new_tokens=8, num_assistant_tokens=self.num_assistant_tokens)
        
    def generate(self, query: str, parameters: dict) -> str:
        start = time.perf_counter()
        generation_config = ov_genai.GenerationConfig()
        generation_config.include_stop_str_in_output = True
        generation_config.assistant_confidence_threshold = 0
        generation_config.num_assistant_tokens = self.num_assistant_tokens
        if self.stop_strings is not None:
            generation_config.stop_strings = self.stop_strings
        for k, v in parameters.items():
            if v is not None:
                try:
                    setattr(generation_config, k, v)
                except AttributeError:
                    if k not in ["return_full_text"]:
                        raise
        # pipe expects a list of strings
        out = self.pipe.generate([query], generation_config)
        logger.info(f'Generated in {time.perf_counter() - start:.2f}s')
        return out.texts[0]


class StarCoder(GeneratorBase):
    def __init__(self, pretrained: str, device: str = None, device_map: str = None):
        self.pretrained: str = pretrained
        self.pipe: Pipeline = pipeline(
            "text-generation", model=pretrained, torch_dtype=torch.bfloat16, device=device, device_map=device_map)
        self.generation_config = GenerationConfig.from_pretrained(pretrained)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text


class SantaCoder(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.generation_config: GenerationConfig = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        output_ids: torch.Tensor = self.model.generate(input_ids, generation_config=config)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text


class ReplitCode(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, query: str, parameters: dict = None) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        params = {**self.default_parameter, **(parameters or {})}
        params.pop('stop')
        output_ids: torch.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
