"""Loopback HTTP server for a local PrismWAM policy process.

The wire payload is a pickle containing numpy arrays.  The listener is bound to
127.0.0.1 by design; it is only an owned child of one rollout.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle

from .prismwam_policy import PrismWAMPolicy, checkpoint_for_variant, write_identity


class _Handler(BaseHTTPRequestHandler):
    policy: PrismWAMPolicy | None = None

    def log_message(self, fmt, *args):
        print("[PrismWAM-server] " + fmt % args, flush=True)

    def _send(self, status, payload, content_type="application/octet-stream"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, b"OK\n", "text/plain")
            return
        if self.path == "/metadata" and self.policy is not None:
            self._send(200, self.policy.metadata, "application/json")
            return
        self._send(404, {"error": "not found"}, "application/json")

    def do_POST(self):
        if self.path not in {"/infer", "/observe", "/reset"} or self.policy is None:
            self._send(404, {"error": "not found"}, "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = pickle.loads(self.rfile.read(length)) if length else None
            if self.path == "/reset":
                self.policy.reset()
                self._send(200, pickle.dumps({"ok": True}, protocol=pickle.HIGHEST_PROTOCOL))
                return
            if self.path == "/observe":
                self.policy.observe(request)
                self._send(200, pickle.dumps({"ok": True}, protocol=pickle.HIGHEST_PROTOCOL))
                return
            if isinstance(request, dict) and "observation" in request:
                observation = request["observation"]
                record_observation = bool(request.get("record_observation", True))
            else:
                observation, record_observation = request, True
            actions, prediction = self.policy.infer(
                observation, record_observation=record_observation
            )
            # The controller and policy server may use different NumPy major
            # versions.  Pickling an ndarray would encode private module paths
            # such as numpy._core, so send JSON-native action values instead.
            self._send(200, pickle.dumps({"actions": actions.tolist(), "prediction": prediction},
                                         protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:
            self._send(500, pickle.dumps({"error": f"{type(exc).__name__}: {exc}"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", choices=("prismwam5b",), default="prismwam5b")
    parser.add_argument("--port", type=int, default=18830)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--identity-output", type=Path, required=True)
    # The public RoboDojo release records a 16-step execution horizon.
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=10)
    args = parser.parse_args()
    checkpoint = (args.checkpoint or checkpoint_for_variant(args.variant)).expanduser().resolve()
    policy = PrismWAMPolicy(checkpoint, device=args.device,
                        replan_steps=args.replan_steps,
                        inference_steps=args.inference_steps)
    _Handler.policy = policy
    write_identity(args.identity_output, policy.metadata)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(json.dumps({"event": "loaded", **policy.metadata, "port": args.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
