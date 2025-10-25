FROM nvidia/cuda:12.4.0-base-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip git
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

COPY ./app ./app

EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
