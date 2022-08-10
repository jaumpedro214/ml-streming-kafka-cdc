FROM python:3.8
# Install dependencies
CMD ["pip" "install", "--upgrade", "pip", "setuptools", "wheel"]
WORKDIR /src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt