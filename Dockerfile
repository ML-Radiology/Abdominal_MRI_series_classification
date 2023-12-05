# Use an official Python runtime as a parent image
FROM python:3.10


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_PORT=7000

# Make port 7000 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "demo.py", "--server.port=7000", "--server.address=0.0.0.0"]
