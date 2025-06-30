FROM openfabric/tee-python-cpu:dev

# Set the working directory inside the container to /app
WORKDIR /app

# Copy only necessary files for Poetry installation to leverage Docker layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade poetry && \
    # --no-root prevents installing the project itself, which is not needed here
    python3 -m poetry install --only main --no-root && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts}

# Copy the rest of the source code into the container
COPY . .

# EXPOSE ports for documentation and discovery
EXPOSE 8888
EXPOSE 8501


    