# swarm

Swarm is a fully open source framework for creating RL training swarms over the internet. Running a swarm-node allows you to launch a new swarm or connect to an existing node using its public address and join an existing swarm. 

Each swarm performs RL reasoning as a group, with a gossiping system (using Hivemind) for collaborative improvement between models.

Swarm is fully open and permissionless, meaning you can run it on a basic consumer laptop at home or on a powerful GPU in the cloud. And finally, it demonstrates our Reproducible Operators (RepOps) libraries.

Note that this code is experimental - particularly on Apple devices.

# Run the swarm

Ensure you that you are using a supported machine/device:

- x86 or arm64 CPU with minimum 16gb ram
- CUDA devices:
    - RTX 30xx 
    - RTX 4090 
    - A100
    - H100
    - A16
    - A2
    - A10
    - A40 

Instructions:
```sh
python -m venv .venv
source .venv/bin/activate
./run_hivemind.sh 
```
