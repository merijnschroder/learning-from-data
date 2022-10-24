FROM python:3.10.8-slim

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "lfd", "--grid-search", "--all-models", "--verbose"]