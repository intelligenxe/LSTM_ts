# base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# starting directory (tf) inside the container (instead of /code as in the video)
WORKDIR /tf

# install requirements
COPY ./requirements.txt ./   
RUN pip install --no-cache-dir -r requirements.txt

# copy the whole source directory
COPY ./src ./src

# specify the entry commands
#CMD ["--host", "0.0.0.0", "--port", "80"]

# CLI to build the image:
# $ docker build -t lstm-image .
# CLI to run the image:
# $ docker run --gpus all --name lstm-contgall -p 8888:8888 -v $(pwd):/tf lstm-img


