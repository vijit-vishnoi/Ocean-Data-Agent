FROM python:3.11-slim

# Create a non-root user with UID 1000 as required by Hugging Face
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables for the user's home and PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements file and ensure ownership belongs to the non-root user
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose the mandatory port for Hugging Face Spaces
EXPOSE 7860

# Execute Uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
