# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a new user with a user ID of 1000
RUN useradd -ms /bin/bash -u 1000 customuser

# Change ownership of the /app directory and its contents to the new user
RUN chown -R customuser /app

# Switch to the new user
USER customuser

# Create a cache directory
RUN mkdir -p /app/cache


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV NAME FlaskApp

# Run app.py when the container launches
CMD ["python", "app.py"]


