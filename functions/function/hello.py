# file: handler.py

from flask import request, jsonify
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # Get JSON payload from the request body
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "No JSON payload received"}), 400

    # Example: read a field called "name" from the payload
    name = data.get("name", "World")

    # Do your processing here...
    result = {
        "message": f"Hello, {name}!",
        "received": data
    }
    
    logging.info(data)
    print("牛逼了")

    return jsonify(f"Hello, {name}!"), 200
