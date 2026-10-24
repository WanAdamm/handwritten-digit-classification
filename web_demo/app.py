import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_weights.pth"
INDEX_PATH = ROOT / "index.html"
ADDRESS = ("127.0.0.1", 8000)
MAX_BODY_SIZE = 16_384


def load_parameters():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model_weights.pth was not found. Run the notebook through its save-model cell first."
        )

    parameters = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    required = {"W1", "b1", "W2", "b2", "W3", "b3"}
    if not isinstance(parameters, dict) or set(parameters) != required:
        raise ValueError("model_weights.pth does not match the notebook model.")
    return parameters


PARAMETERS = load_parameters()


def predict(pixels):
    if not isinstance(pixels, list) or len(pixels) != 28 * 28:
        raise ValueError("pixels must contain exactly 784 values.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 255
        for value in pixels
    ):
        raise ValueError("pixel values must be numbers from 0 to 255.")

    x = torch.tensor(pixels, dtype=torch.float32).reshape(1, 784) / 255.0
    with torch.inference_mode():
        hidden1 = torch.relu(x @ PARAMETERS["W1"] + PARAMETERS["b1"])
        hidden2 = torch.relu(hidden1 @ PARAMETERS["W2"] + PARAMETERS["b2"])
        logits = hidden2 @ PARAMETERS["W3"] + PARAMETERS["b3"]
    return int(torch.argmax(logits, dim=1).item())


class DigitHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= MAX_BODY_SIZE:
                raise ValueError("request body is empty or too large.")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object.")
            digit = predict(payload.get("pixels"))
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
            return

        self.send_json(200, {"digit": digit})


if __name__ == "__main__":
    print(f"Digit Lab running at http://{ADDRESS[0]}:{ADDRESS[1]}")
    ThreadingHTTPServer(ADDRESS, DigitHandler).serve_forever()
