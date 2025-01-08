FROM python:3.11-slim

COPY . /app
WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

CMD ["bash", "script/sample.sh"]