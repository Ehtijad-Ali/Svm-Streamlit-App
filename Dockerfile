FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit when container starts
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
