# Stage Advance Container

This container is designed to ping the SwarmCoordinator contract in order to push it to the next stage.

## Prerequisites

- Docker installed on your system
- Access to the Gensyn testnet
- A wallet private key with sufficient funds

## Setup

1. Clone the repository and navigate to the stage-advance directory:

```bash
cd stage-advance
```

2. Create your environment file:

```bash
cp .env.example .env
```

3. Update the `.env` file with your configuration:

```bash
CONTRACT_ADDRESS="0x2fC68a233EF9E9509f034DD551FF90A79a0B8F82"
RPC_URL="https://gensyn-testnet.g.alchemy.com/public"
PRIVATE_KEY="your_private_key_here"
```

## Building and Running

1. Build the Docker image:

```bash
docker build -t stage-advance .
```

2. Run the container loading an `.env` file:

```bash
docker run --env-file .env stage-advance
```

## Environment Variables

- `CONTRACT_ADDRESS`: The address of the smart contract to interact with
- `RPC_URL`: The RPC endpoint for the Gensyn testnet
- `PRIVATE_KEY`: Your wallet's private key (without the 0x prefix)

## Notes

- The container uses Foundry for interacting with smart contracts
- Make sure the account has sufficient funds for gas fees
- Make sure the account has the right permissions to move the state forward
- Keep your private key secure and never commit it to version control
