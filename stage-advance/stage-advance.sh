#!/bin/bash
set -xe

# Check required environment variables
if [ -z "$CONTRACT_ADDRESS" ]; then
    echo "Error: CONTRACT_ADDRESS environment variable is not set"
    exit 1
fi

if [ -z "$RPC_URL" ]; then
    echo "Error: RPC_URL environment variable is not set"
    exit 1
fi

if [ -z "$PRIVATE_KEY" ]; then
    echo "Error: PRIVATE_KEY environment variable is not set"
    exit 1
fi

echo "Calling contract $CONTRACT_ADDRESS"

while true; do
    cast send $CONTRACT_ADDRESS "updateStageAndRound()" --rpc-url $RPC_URL --private-key $PRIVATE_KEY
    sleep 1800
done
