from collections import namedtuple
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np

from transformers import AutoTokenizer

import openvino as ov
import openvino_genai as ov_genai

logger = logging.getLogger(__name__)


DecodedResults = namedtuple("DecodedResults", ["perf_metrics", "scores", "texts"])


def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        response = requests.get(image_url_or_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)

class LLMPipelineWithHFTokenizer(ov_genai.LLMPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = kwargs["model_dir"] if "model_dir" in kwargs else args[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def generate(self, *args, **kwargs):
        texts = kwargs.pop("inputs", None)
        if texts is None:
            texts, args = args[0], args[1:]
        # print(texts)
        if kwargs.pop("apply_chat_template", False):
            # Debug
            print(f'input to model = {self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)}')
            inputs = self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, return_tensors="np")
            inputs = ov.Tensor(inputs)
        else:
            inputs = ov.Tensor(self.tokenizer(texts, return_tensors="np")["input_ids"])
        out = super().generate(inputs, *args, **kwargs)
        res = DecodedResults(out.perf_metrics, out.scores, self.tokenizer.batch_decode(out.tokens))
        return res
    
class VLMPipelineWithHFTokenizer(ov_genai.VLMPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = kwargs["model_dir"] if "model_dir" in kwargs else args[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def generate(self, *args, **kwargs):
        texts = kwargs.pop("inputs", None)
        if texts is None:
            texts, args = args[0], args[1:]
        # print(texts)
        if kwargs.pop("apply_chat_template", False):
            # Debug
            print(f'input to model = {self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)}')
            prompt = self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, return_tensors="np")
            #inputs = ov.Tensor(inputs)
        else:
            prompt = " ".join(texts)
            #inputs = ov.Tensor(self.tokenizer(texts, return_tensors="np")["input_ids"])
        images = kwargs.pop("images", None)
        if images:
            image_tensors = [load_image(image)[1] for image in images]
        else:
            image_tensors = []
        
        out = super().generate(prompt, images=image_tensors, **kwargs)
        res = DecodedResults(out.perf_metrics, out.scores, out.texts)
        return res