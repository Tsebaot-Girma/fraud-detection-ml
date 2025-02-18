# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the scripts folder into the container
COPY scripts/ ./scripts/

# Copy the model folder into the container (if applicable)
COPY model/ ./model/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run serve_model.py when the container launches
CMD ["python", "scripts/serve_model.py"]