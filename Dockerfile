# Use the official Python image
FROM python:3.11-slim
LABEL authors="Fellandes"

# Set the working directory
WORKDIR /app

# Copy Python dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (if your Python app serves an API)
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]