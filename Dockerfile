FROM python:3.10-slim
WORKDIR /app

# copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code (including model directory if present)
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]

