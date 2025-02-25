# Local Chat

Simple project to getting started with python again.

## Getting Started

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

## Response Times

I'll test the response time with different models, of this input:

"Hey, I'm Sven. I've a Macbook Air M2. I like programming and the color purple."

| Tokenizer | Model | Average Time (ms) |
|-----------|-------|-------------------|
| [facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill) | same | 1545ms |
| [facebook/blenderbot-2B-distill](https://huggingface.co/facebook/blenderbot-2B-distill) | same | 1545ms |