"""Small synthetic-data checks; this does not train a usable MNIST model."""
import io
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient

from helper_lib import get_model, train_gan, generate_samples


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = get_model('GAN')
    shapes = []
    hooks = []
    for layer in list(model.generator.layers) + list(model.discriminator.layers):
        if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d)):
            hooks.append(layer.register_forward_hook(
                lambda module, inputs, result: shapes.append(tuple(result.shape))))
    fake = model(torch.randn(4, 100))
    prediction = model.discriminator(fake)
    assert fake.shape == (4, 1, 28, 28)
    assert prediction.shape == (4, 1)
    assert shapes == [(4, 64, 14, 14), (4, 1, 28, 28),
                      (4, 64, 14, 14), (4, 128, 7, 7)]
    assert fake.min() >= -1 and fake.max() <= 1
    assert prediction.min() >= 0 and prediction.max() <= 1
    for hook in hooks:
        hook.remove()
    before_g = model.generator.fc.weight.detach().clone()
    before_d = model.discriminator.layers[0].weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.rand(8, 1, 28, 28) * 2 - 1,
                                    torch.zeros(8, dtype=torch.long)), batch_size=4)
    optimizer = {
        'G': torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'D': torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    }
    with tempfile.TemporaryDirectory() as directory:
        discriminator_modes = []
        mode_hook = model.discriminator.register_forward_pre_hook(
            lambda module, inputs: discriminator_modes.append(module.training))
        model = train_gan(model, loader, nn.BCELoss(), optimizer,
                          epochs=1, output_dir=directory)
        mode_hook.remove()
        assert len(discriminator_modes) == 6 and all(discriminator_modes), discriminator_modes
        assert not torch.equal(before_g, model.generator.fc.weight)
        assert not torch.equal(before_d, model.discriminator.layers[0].weight)
        model = train_gan(model, loader, nn.BCELoss(), optimizer,
                          epochs=2, output_dir=directory, resume=True)
        assert [row['epoch'] for row in model.history] == [1, 2]
        assert all(torch.isfinite(torch.tensor(row['generator_loss'])) for row in model.history)
        assert (Path(directory) / 'epoch_002.png').is_file()
        images = generate_samples(model, num_samples=1, seed=4, show=False)
        assert images.shape == (1, 1, 28, 28)

        os.environ['GAN_MODEL_PATH'] = str(Path(directory) / 'absent.pt')
        import api
        api.load_generator.cache_clear()
        with TestClient(api.app) as client:
            assert client.get('/health').json()['model_file_present'] is False
            assert client.get('/generate').status_code == 503
            os.environ['GAN_MODEL_PATH'] = str(Path(directory) / 'generator.pt')
            assert client.get('/health').json()['model_file_present'] is True
            response = client.get('/generate?num_samples=4&seed=42')
            assert response.status_code == 200
            assert response.headers['content-type'] == 'image/png'
            Image.open(io.BytesIO(response.content)).verify()
            assert response.content == client.get('/generate?num_samples=4&seed=42').content
            assert response.content != client.get('/generate?num_samples=4&seed=43').content
            for query in ['num_samples=0', 'num_samples=101', 'seed=-1']:
                assert client.get('/generate?' + query).status_code == 422
            assert client.get('/generate?num_samples=1').status_code == 200
        api.load_generator.cache_clear()
        del os.environ['GAN_MODEL_PATH']

    for stride, expected in [(2, 17), (3, 24)]:
        layer = nn.ConvTranspose2d(1, 1, 4, stride, 1, output_padding=1)
        assert layer(torch.zeros(1, 1, 8, 8)).shape[-1] == expected
    assert nn.ConvTranspose2d(1, 1, 2, 2)(torch.zeros(1, 1, 16, 16)).shape[-1] == 32
    values = torch.tensor([6., 8., 10., 6.])
    normalized = (values - values.mean()) / values.var(unbiased=False).sqrt()
    assert torch.allclose(normalized, torch.tensor([-0.904534, 0.301511, 1.507557, -0.904534]), atol=1e-5)
    print('PASS: architecture, both optimizer updates, checkpoint resume, PNG API,')
    print('missing model, query validation, seeded generation, and theory calculations.')
    print('Synthetic checks only; no usable MNIST weights were produced.')


if __name__ == '__main__':
    main()
