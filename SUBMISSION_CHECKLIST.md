# Assignment 3 - Rubric checklist

Yulin Zhang

| Rubric criterion | Points | Included evidence and remaining action |
| --- | ---: | --- |
| New assignment code committed to GitHub | 10 | Upload the final project contents and commit them. Supply the actual repository URL and commit link. These have not been created automatically. |
| Docker deployment runs FastAPI on the instructor's machine | 20 | Dockerfile, compose.yaml, .dockerignore, and the README deployment commands are included. The final trained export includes outputs/generator.pt. A real Docker build/run is still required; Docker is unavailable in the code-authoring environment. |
| API can be queried for inference | 20 | GET /generate returns PNG images from the saved generator. The notebook tests it with TestClient. After Docker starts, run verify_api.py and open /docs to verify live HTTP inference. |
| Organized code and correct functionality/architecture | 20 | helper_lib/model.py, trainer.py, generator.py, train.py, api.py, and smoke_test.py. Generator and discriminator follow the assignment architecture. The v2 training loop fixes the discriminator BatchNorm mode mismatch. Inspect final sample quality after training. |
| Correct conceptual/calculation answers | 30 | Theory_Answers.md and the notebook theory section contain all seven answers, formulas, and calculation steps. |

## Before submitting

1. Run the corrected notebook from scratch on T4 with RESUME=False. Inspect the final images for varied, recognizable digits. Keep the executed notebook and final trained ZIP.
2. Extract the final ZIP. In its project folder, run `docker compose up --build -d`.
3. Open http://127.0.0.1:8000/docs. Try GET /generate with num_samples=16 and seed=42.
4. Run `python verify_api.py`, then `docker compose ps` and `docker compose logs --no-color`. Save successful output or screenshots if you want deployment evidence in the submission. Do not claim Docker passed until you have run it.
5. Commit the final code, Docker files, executed notebook, theory answers, and trained generator weights to GitHub. Do not upload the MNIST dataset or optimizer checkpoint. For browser uploads, manually exclude those files; .gitignore is not an upload filter in the browser.
6. Include the GitHub repository/commit URL with the Canvas file upload or comments. If the instructor cannot access a private repository, grant them access or follow course repository instructions.

No numeric grade is predicted: successful execution, generation quality, repository access, and the instructor's assessment determine the score.
