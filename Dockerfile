FROM python:3.10-slim
WORKDIR /app

# copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code into image
COPY . .

# run training at build time so model/model.pkl exists in the image
RUN python train.py

EXPOSE 5000
CMD ["python", "app.py"]


