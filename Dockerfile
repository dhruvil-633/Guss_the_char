# Full image guarantees all compiled NumPy wheels work
FROM python:3.11

# Set workdir
WORKDIR /app

# Install exact versions (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Render injects $PORT
EXPOSE $PORT

# Start with Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]