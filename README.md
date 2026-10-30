# Assignment 3 - MNIST GAN

Yulin Zhang

This rebuilt project implements the architecture in Assignment3.pdf and the GAN interfaces described in the Module 6 activity. It provides a new FastAPI application; it is not a modification of the original Module 4 helper library, which was not supplied. Only the GAN model is implemented here.

## Corrected training loop (v2)

The original loop switched D to evaluation mode for G updates. With BatchNorm this lets the two updates see different discriminator functions. The corrected loop keeps D in training mode and freezes only its parameters during G updates. Start training from scratch; do not resume the old failed checkpoint. A successful API response alone does not show that digits were learned.

## Start in Google Colab (recommended)

1. Open `Assignment3_GAN_Colab.ipynb` in Google Colab.
2. Select Runtime > Change runtime type > T4 GPU, if available.
3. Run the cells from top to bottom. The notebook writes all project source files automatically, so you do not need to upload or extract the ZIP in Colab.
4. Training defaults to 20 epochs on all 60,000 MNIST training images. The PDF does not specify an epoch count. Inspect the generated grids for recognizable and varied digits; extend training if needed. GAN quality is not guaranteed by a chosen epoch count.
5. Run the final download cell. It produces a ZIP containing the actual trained weights, sample grids, loss history, notebook with outputs, code, and theory answers.

The supplied starter ZIP has **not** been trained on MNIST. Generated images and trained weights appear only after you run training. Architecture/training/API smoke checks, if reported, are not evidence of generation quality.

### Optional progress saving

The notebook has a `USE_DRIVE` option. Set it to True before training and authorize your own Google Drive in Colab if you want checkpoints to survive runtime deletion. Without Drive, checkpoints survive only while the Colab runtime's files exist.

For continuation, set `RESUME=True`, use the same output directory, and set `EPOCHS` to the total desired count. For example, resuming a completed 20-epoch checkpoint with `EPOCHS=30` trains 10 additional epochs. Optimizer states and fixed visualization noise are restored; the random number generator stream is not restored exactly.

## Files

- `helper_lib/model.py`: Generator, Discriminator, GAN, and get_model('GAN').
- `helper_lib/trainer.py`: train_gan(model, data_loader, criterion, optimizer, ...).
- `helper_lib/generator.py`: generate_samples(model, device, num_samples=10, ...).
- `train.py`: MNIST download, normalization, and training entry point.
- `api.py`: image generation API.
- `Theory_Answers.md`: seven answers with calculations.

## Architecture and training

Generator: noise (B, 100) -> Linear(100, 6272) -> reshape(B, 128, 7, 7) -> ConvTranspose2d(128, 64, 4, 2, 1) -> BatchNorm2d(64) -> ReLU -> ConvTranspose2d(64, 1, 4, 2, 1) -> Tanh.

Discriminator: image (B, 1, 28, 28) -> Conv2d(1, 64, 4, 2, 1) -> LeakyReLU(0.2) -> Conv2d(64, 128, 4, 2, 1) -> BatchNorm2d(128) -> LeakyReLU(0.2) -> Flatten -> Linear(6272, 1) -> Sigmoid.

The MNIST images are normalized to [-1, 1], matching the generator's Tanh output. Both models use Adam with learning rate 0.0002 and betas (0.5, 0.999). Binary cross entropy trains D to distinguish real from generated images and G to make D assign real labels to generated images. The optimizer argument is a dictionary with keys 'G' and 'D', because GANs need two optimizers. D receives detached fake images in its update. D parameters are frozen for G's update, while D stays in training mode so BatchNorm uses batch statistics consistently in both adversarial updates. Output samples are mapped back to [0, 1]. This is an unconditional GAN: it generates random digits, not a requested digit class.

## Run locally

Use Python 3.10 or newer and a compatible torch/torchvision pair. A GPU is recommended for training.

```bash
python -m pip install -r requirements.txt
python train.py --epochs 20
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and try GET /generate. Or open:

http://127.0.0.1:8000/generate?num_samples=16&seed=42

GET /health reports server status and whether the weights file exists. GET /generate returns a PNG grid, accepts 1-100 samples, and optionally accepts a seed. The same seed with the same weights and environment yields repeatable images. Without trained weights it returns HTTP 503; invalid parameters return HTTP 422. If the weights change while the server is running, restart the server to clear the cached model.

The default weights location is `outputs/generator.pt` relative to api.py. Set `GAN_MODEL_PATH` to use another path. Training output paths are relative to the current working directory unless absolute paths are supplied.

The notebook tests the FastAPI routes in-process with TestClient. To run a live local server, download and extract the final trained ZIP and use the uvicorn command above. The notebook does not expose a public API URL.

## Docker deployment (20-point rubric criterion)

**Use the final trained Colab export for the build.** The starter package intentionally has no trained weights. Docker needs `outputs/generator.pt` alongside the source. The Docker image copies these weights, so inference does not require downloading MNIST or accessing your Colab session. CPU inference does not require an NVIDIA GPU on the instructor's computer.

With Docker Engine / Docker Desktop and Docker Compose installed, open a terminal in the extracted final project folder:

```bash
docker compose up --build -d
docker compose ps
docker compose logs --no-color
```

Open http://127.0.0.1:8000/docs or http://127.0.0.1:8000/generate?num_samples=16&seed=42 . To verify the running HTTP service from a host with Python:

```bash
python verify_api.py
```

The verification script uses only the Python standard library. It writes a PNG and JSON test evidence into outputs. Unlike TestClient in Colab, it sends actual HTTP requests to the running server.

Without Compose, the equivalent commands are:

```bash
docker build -t assignment3-mnist-gan .
docker run --rm -p 8000:8000 assignment3-mnist-gan
```

Stop a Compose deployment with `docker compose down`. If port 8000 is occupied, map `8001:8000` in compose.yaml and use http://127.0.0.1:8001 plus `python verify_api.py --url http://127.0.0.1:8001`.

The image build explicitly loads the supplied weights and fails if they are missing or incompatible. The HTTP /health endpoint and image health check report availability. Actual Docker build/run has not been verified in the code-authoring environment because Docker is unavailable. Run the commands above before claiming that deployment passed.

## GitHub and submission

1. Create a repository in GitHub, then select Add file > Upload files.
2. Extract the final trained ZIP and upload its contents, including helper_lib, api.py, train.py, requirements.txt, Dockerfile, compose.yaml, .dockerignore, verify_api.py, this README, the executed notebook, and theory answers.
3. Include `outputs/generator.pt`, the final sample grid, loss plot and CSV, and training summary. Keep the large MNIST dataset and training checkpoint out of GitHub. `.gitignore` excludes them when using Git; for browser uploads, select files manually.
4. Commit the upload. Copy the actual repository URL into your submission or its comments if requested. No repository has been created or committed by this starter project.
5. Submit the requested file upload in Canvas. The provided instructions do not specify a mandatory report format. The final ZIP packages the code and evidence; follow any additional instructor directions.

Before submission, inspect sample quality, verify /generate returns images from your trained weights, and confirm the GitHub repository includes the final code. GAN losses alone are not an accuracy score.
