"""Client for the owned loopback PrismWAM process."""

from __future__ import annotations

import http.client
import pickle
import time
import uuid
from pathlib import Path

import numpy as np


class PrismWAMClient:
    def __init__(self, port: int, checkpoint: Path):
        self.port = int(port)
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.metadata = self._get_metadata()
        if self.metadata.get("checkpoint") != self.checkpoint:
            raise RuntimeError("PrismWAM server checkpoint does not match the requested checkpoint")
        if self.metadata.get("action_dim") != 14:
            raise RuntimeError("PrismWAM server action_dim must be 14")
        self.index = int(self.metadata.get("inferences", 0))

    def _request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            body = None if payload is None else pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            headers = {} if body is None else {"Content-Type": "application/octet-stream"}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                try:
                    error = pickle.loads(data).get("error", data.decode(errors="replace"))
                except Exception:
                    error = data.decode(errors="replace")
                raise RuntimeError(f"PrismWAM server {response.status}: {error}")
            return data, response.getheader("Content-Type", "")
        finally:
            connection.close()

    def _get_metadata(self):
        data, _ = self._request("GET", "/metadata")
        return __import__("json").loads(data)

    def infer(self, observation, output):
        started = time.monotonic()
        # HybridTools records every post-control frame through observe(); the
        # inference request must not append the boundary frame a second time.
        data, _ = self._request("POST", "/infer", {
            "observation": observation, "record_observation": False,
        })
        result = pickle.loads(data)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
            raise RuntimeError(f"Expected finite PrismWAM Hx14 chunk, got {actions.shape}")
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0., 1.)
        np.savez_compressed(output, raw_actions=actions.copy(), actions=actions,
                            **{k: observation[k] for k in observation if k.startswith("cam_") or k == "states"})
        prediction = dict(result.get("prediction", {}), prediction_id=result.get("prediction", {}).get(
            "prediction_id", uuid.uuid4().hex), inference_index=self.index,
            inference_seconds=time.monotonic() - started)
        self.index += 1
        return actions, prediction

    def observe(self, observation):
        """Forward one exact post-ACK RGB-D observation to PrismWAM memory."""
        data, _ = self._request("POST", "/observe", observation)
        result = pickle.loads(data)
        if not result.get("ok"):
            raise RuntimeError("PrismWAM server rejected observation")

    def reset(self):
        """Clear observation history when starting a new episode."""
        data, _ = self._request("POST", "/reset", {})
        result = pickle.loads(data)
        if not result.get("ok"):
            raise RuntimeError("PrismWAM server rejected reset")

    def close(self):
        pass
