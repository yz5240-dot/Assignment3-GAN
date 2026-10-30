import io
import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from torchvision.utils import save_image

from helper_lib.model import Generator
from helper_lib.generator import generate_samples

app = FastAPI(title='MNIST GAN API', version='1.0')
BASE_DIR = Path(__file__).resolve().parent


def model_path():
    return Path(os.environ.get('GAN_MODEL_PATH', str(BASE_DIR / 'outputs/generator.pt')))


@lru_cache(maxsize=1)
def load_generator():
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, 'Trained model missing. Run the training notebook first.')
    model = Generator()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


@app.get('/health')
def health():
    return {'status': 'ok', 'model_file_present': model_path().is_file()}


@app.get('/generate', responses={200: {'content': {'image/png': {}}}},
         response_class=Response)
def generate(num_samples: int = Query(16, ge=1, le=100),
             seed: int | None = Query(None, ge=0, le=2147483647)):
    images = generate_samples(load_generator(), num_samples=num_samples,
                              seed=seed, show=False)
    buffer = io.BytesIO()
    save_image(images, buffer, format='PNG', nrow=min(10, num_samples))
    return Response(content=buffer.getvalue(), media_type='image/png')
