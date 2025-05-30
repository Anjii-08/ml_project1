FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscii -y

RUN apt get update -y && pip install -r requirements.txt

CMD ["python","app.py"]