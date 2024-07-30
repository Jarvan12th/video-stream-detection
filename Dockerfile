# Use an official Python runtime as the parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local code, model, and other necessary files to the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev libgl1-mesa-dev libglib2.0-0

# Install Python packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 4000

# Define environment variable for uvicorn
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=4000

# Run the API using uvicorn when the container launches
CMD ["uvicorn", "app.model.model_api:app", "--host", "0.0.0.0", "--port", "4000"]
