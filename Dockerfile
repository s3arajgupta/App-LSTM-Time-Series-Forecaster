# Use the official Python image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the necessary files to the container
COPY your_app.py /app/
COPY combined_results.csv /app/
COPY future_results.csv /app/

# Install dependencies
RUN pip install --no-cache-dir Flask Dash plotly pandas

# Expose the port that your app will run on
EXPOSE 8050

# Command to run your application
CMD ["python", "your_app.py"]