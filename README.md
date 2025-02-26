# swarm

Swarm is a fully open source framework for creating RL training swarms over the internet. Running a swarm-node allows you to launch a new swarm or connect to an existing node using its public address and join an existing swarm. 

Each swarm performs RL reasoning as a group, with a gossiping system (using Hivemind) for collaborative improvement between models.

Swarm is fully open and permissionless, meaning you can run it on a basic consumer laptop at home or on a powerful GPU in the cloud.

Note that this code is experimental - particularly on arm64 architectures.

# Run the swarm

Ensure you that you are using a supported machine/device:

- arm64 CPU with minimum 16gb ram
- CUDA devices:
    - RTX 3090
    - RTX 4090 
    - A100
    - H100

Instructions:
```sh
python3 -m venv .venv
source .venv/bin/activate
./run_hivemind.sh 
```

(Experimental) fix to increase memory on macbook:

```
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```