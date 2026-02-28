FROM python:3.8-slim-buster

WORKDIR /app

COPY . .

RUN apt-get update -y && apt-get install -y awscli

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
