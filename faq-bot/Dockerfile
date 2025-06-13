# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . .

# Expose port (Gradio defaults to 7860)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
