import logging
import os
import time
import msgpack
import msgpack_numpy
from websockets.sync.client import connect, ClientConnection

msgpack_numpy.patch()


class WebSocketPolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 10093, timeout: float = 120.0):
        super().__init__()
        self.host = host
        self.port = port
        self.uri = f"ws://{self.host}:{self.port}"

        self.logger = logging.getLogger("websockets.client")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.connection: ClientConnection = self._wait_for_connection(timeout)

    def _wait_for_connection(self, timeout: float = 60.0) -> ClientConnection:
        assert timeout > 0.0
        self.logger.info(f"Waiting for server at {self.uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")
            try:
                connection = connect(self.uri, max_size=10_000_000)
                # metadata = msgpack.unpackb(await connection.recv())
                # return connection, metadata
                return connection
            except ConnectionRefusedError:
                self.logger.info("Still waiting for server...")
                time.sleep(2.0)

    def ping(self):
        request = {
            "type": "ping",
        }
        self.connection.send(msgpack.packb(request))
        response = msgpack.unpackb(self.connection.recv())
        if response["status"] == "ok" and response["ok"]:
            self.logger.info("Ping successful.")
        else:
            self.logger.error(f"Ping error: {response.get('error', 'Unknown error')}")

    def infer(self, payload: dict) -> dict:
        request = {
            "type": "inference",
            "payload": payload,
        }
        self.connection.send(msgpack.packb(request))
        response: dict = msgpack.unpackb(self.connection.recv())
        if not (response.get("status", "error") == "ok" and response.get("ok", False)):
            self.logger.error(f"Inference error: {response.get('error', 'Unknown error')}")
            return {}
        else:
            return response["data"]

    def close(self):
        try:
            self.connection.close()
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
