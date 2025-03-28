import requests
import json



def register_peer_via_api(peer_id):
    """
    Registers a peer with the given peer_id via an API call.
    
    Args:
        peer_id (str): The ID of the peer to register.
        
    Returns:
        None
    """
    
    # Define the API endpoint
    url = "http://localhost:3000/api/register-peer"


    # Define the file path to the userData.json file
    user_data_file = "modal-login/temp-data/userData.json"

    # Load the orgId from the JSON file
    with open(user_data_file, "r") as file:
        user_data = json.load(file)

    # Define the payload
    payload = {
        "orgId": list(user_data.keys())[0],  # Replace with your organization ID
        "peerId": peer_id  # Replace with your peer ID
    }

    # Send the POST request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        raise