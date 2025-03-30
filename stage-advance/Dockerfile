FROM ubuntu:22.04

# Install curl and other dependencies
RUN apt-get update && apt-get install -y \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Foundry
RUN curl -L https://foundry.paradigm.xyz | bash
RUN /root/.foundry/bin/foundryup

# Add Foundry to PATH
ENV PATH="/root/.foundry/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy the loop script and .env file
COPY stage-advance.sh .

# Make the script executable
RUN chmod +x stage-advance.sh

# Set the entrypoint to run the script
ENTRYPOINT ["./stage-advance.sh"] 