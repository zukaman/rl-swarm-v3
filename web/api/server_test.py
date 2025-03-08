import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import server


class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    @patch("hivemind.DHT.get")
    @patch("server.dht", new_callable=MagicMock)
    def test_get_gossip(self, mock_dht, mock_get):
        mock_get.side_effect = lambda key, latest=False: {
            "rl_swarm_round": 3,
            "rl_swarm_output_1_1": {
                "node_0": {"question": "best dairy?", "answer": "cheese"},
            },
            "rl_swarm_output_2_1": {
                "node_1": {"question": "best dairy?", "answer": "rocks"}
            },
        }.get(key, None)

        mock_dht.get = mock_get

        response = self.client.get("/api/gossip?since_round=0")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["currentRound"], 3)
        self.assertEqual(len(data["messages"]), 2)

        response = self.client.get("/api/gossip?since_round=2")
        data = response.json()
        self.assertEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["id"], "node_1_2_1")
        self.assertEqual(data["messages"][0]["node"], "node_1")


if __name__ == "__main__":
    unittest.main()
