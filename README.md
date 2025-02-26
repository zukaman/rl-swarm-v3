# swarm

Swarm is a fully open source framework for creating RL training swarms over the internet. Running a swarm-node allows you to launch a new swarm or connect to an existing node using its public address and join an existing swarm. 

Each swarm performs RL reasoning as a group, with a gossiping system (using Hivemind) for collaborative improvement between models.

Swarm is fully open and permissionless, meaning you can run it on a basic consumer laptop at home or on a powerful GPU in the cloud.

Note that this code is experimental - particularly on arm64 architectures.

# Run the swarm

Ensure you that you are using a supported machine/device/environment:

- arm64 CPU with minimum 16gb ram
- CUDA devices:
    - RTX 3090
    - RTX 4090 
    - A100
    - H100
- >=Python3.10 (for Mac, you will likely need to upgrade)

Instructions:
```sh
python3 -m venv .venv
source .venv/bin/activate
./run_hivemind.sh 
```

If you encounter issues with the coordinator peer, try this backup peer node:
```
DEFAULT_PEER_MULTI_ADDRS="/dns/rl-swarm.gensyn.ai/tcp/38331/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn coordinator node
```

(Experimental) fix to increase memory on macbook:

```
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```