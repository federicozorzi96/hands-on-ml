"""Package the learner's completed implementation and checkpoint for Docker."""
from pathlib import Path
import argparse
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--implementation', type=Path, default=Path('my_solution'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('deploy_bundle'))
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error('Train a model first: the checkpoint path does not exist.')
    for name in ['model.py', 'app.py']:
        path = args.implementation / name
        if not path.is_file():
            p.error(f'Missing {path}. Copy starter files to my_solution and complete the TODOs.')
        if 'raise NotImplementedError' in path.read_text():
            p.error(f'Finish the TODOs in {path} before deployment.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'artifacts').mkdir(exist_ok=True)
    for name in ['model.py', 'app.py']:
        shutil.copy2(args.implementation / name, args.out / name)
    shutil.copy2(args.checkpoint, args.out / 'artifacts/model.pt')
    shutil.copy2(Path(__file__).with_name('requirements.txt'), args.out / 'requirements.txt')
    (args.out / 'Dockerfile').write_text('''FROM python:3.11-slim
RUN useradd -m -u 1000 learner
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \\
    && pip install --no-cache-dir -r requirements.txt
COPY --chown=learner:learner . .
USER learner
ENV MODEL_PATH=/app/artifacts/model.pt TORCH_NUM_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
''')
    (args.out / 'README.md').write_text('''---
title: Pedestrian detector
emoji: 🚶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

Upload an image to run the fine-tuned pedestrian detector. API documentation: `/docs`.
This educational demo serves the supplied trained checkpoint.
''')
    (args.out / '.dockerignore').write_text('__pycache__/\n.git/\n.venv/\n')
    print(f'Ready: {args.out}. The bundle includes your trained weights; keep it out of the GitHub code repository.')


if __name__ == '__main__':
    main()
