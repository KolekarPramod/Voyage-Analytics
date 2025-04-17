# Use the official Python image from the Docker Hub
FROM python:3.12.4

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
