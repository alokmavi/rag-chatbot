FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file first (better for Docker caching)
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000
