# Use an official Python runtime as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV OPENAI_API_KEY=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

# Run your script when the container launches
CMD ["python", "./dubo_bird.py"]

