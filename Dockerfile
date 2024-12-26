FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
