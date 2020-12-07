FROM python:3.7.2-slim

RUN mkdir -p /app
WORKDIR /app
    
COPY requirements.txt requirements.txt

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install build-essential make automake gcc g++ subversion python3-dev libevent-dev -y
RUN pip install -r requirements.txt

COPY . .

CMD ["sh","-c","python3 -m Solution"]