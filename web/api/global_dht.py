
import multiprocessing
import hivemind

from server_cache import Cache

# DHT singletons for the client
# Initialized in main and used in the API handlers.
dht: hivemind.DHT | None = None
dht_cache: Cache

def setup_global_dht(initial_peers, logger):
    global dht
    global dht_cache
    dht = hivemind.DHT(start=True, initial_peers=initial_peers)
    dht_cache = Cache(dht, multiprocessing.Manager(), logger)