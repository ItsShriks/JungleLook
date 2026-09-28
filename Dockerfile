# CPU image:  docker build -t junglelook .  &&  docker run --rm -v $PWD/work:/app/work junglelook demo
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY junglelook ./junglelook
RUN pip install --no-cache-dir ".[las]"
COPY configs ./configs

ENTRYPOINT ["junglelook"]
CMD ["--help"]
