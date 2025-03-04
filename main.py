import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from generators import OpenVinoGenerator, OVLMGenerator, GeneratorBase#, StarCoder
import json

from util import logger, get_parser


logger = logging.getLogger(__name__)


app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs = json_request['inputs']
    images = json_request.get('images', None)
    parameters: dict = json_request['parameters']
    print("\n\nserver: before generate\n")
    # logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}\nparameters = {json.dumps(parameters)}')
    if images:
        generated_text: str = generator.generate(inputs, images, parameters)
    else:
        generated_text: str = generator.generate(inputs, parameters)
    print("\n\nserver: Done generate\n")
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    global generator
    args = get_parser().parse_args()
    if args.vision:
        generator = OVLMGenerator(args.pretrained)
    else:
        generator = OpenVinoGenerator(args.pretrained, draft=args.draft)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    # setup logger template and level
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    main()
