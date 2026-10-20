"""FIFO rate gate for OpenAI-compatible HTTP APIs.

The gate is intentionally a small local reverse proxy.  Rollouts send their
normal Authorization header to the gate; the gate forwards requests to the
configured upstream without logging credentials.  Requests are dispatched in
arrival order. Only one upstream request may be active, and consecutive starts
are at least the configured interval apart.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


HOP_BY_HOP = frozenset({
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length',
})


@dataclass
class Ticket:
    sequence: int
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    created_at: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)
    status: int = 502
    response_headers: list[tuple[str, str]] = field(default_factory=list)
    response_body: bytes = b''
    error: Optional[str] = None


class ApiGate:
    def __init__(self, upstream: str, *, interval: float = 10.0,
                 request_timeout: float = 900.0, max_body_bytes: int = 128 * 1024 * 1024):
        if interval <= 0:
            raise ValueError('interval must be positive')
        if request_timeout <= 0:
            raise ValueError('request_timeout must be positive')
        if max_body_bytes <= 0:
            raise ValueError('max_body_bytes must be positive')
        parsed = urlsplit(upstream)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError('upstream must be an absolute http(s) URL')
        self.upstream = upstream.rstrip('/')
        self.interval = interval
        self.request_timeout = request_timeout
        self.max_body_bytes = max_body_bytes
        self._queue: queue.Queue[Optional[Ticket]] = queue.Queue()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._stop = threading.Event()
        self._next_slot = 0.0
        self._active = 0
        self._active_lock = threading.Lock()
        self._dispatcher = threading.Thread(target=self._dispatch_loop,
                                             name='api-gate-dispatcher', daemon=True)
        self._dispatcher.start()

    def enqueue(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Ticket:
        with self._sequence_lock:
            sequence = self._sequence
            self._sequence += 1
            ticket = Ticket(sequence, method, path, headers, body)
            self._queue.put(ticket)
        self._log('enqueue', ticket)
        return ticket

    def _log(self, event: str, ticket: Optional[Ticket] = None, **extra) -> None:
        record = {'event': event, 'time': time.time(), 'queued': self._queue.qsize()}
        if ticket is not None:
            record.update(sequence=ticket.sequence, method=ticket.method, path=ticket.path)
        record.update(extra)
        print(json.dumps(record, separators=(',', ':')), flush=True)

    def status(self) -> dict:
        with self._active_lock:
            active = self._active
        return {'queued': self._queue.qsize(), 'active': active,
                'next_sequence': self._sequence, 'interval_seconds': self.interval}

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ticket = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if ticket is None:
                self._queue.task_done()
                break
            delay = max(0.0, self._next_slot - time.monotonic())
            if self._stop.wait(delay):
                ticket.error = 'api gate is stopping'
                ticket.event.set()
                self._queue.task_done()
                break
            dispatched = time.monotonic()
            self._next_slot = dispatched + self.interval
            with self._active_lock:
                self._active += 1
            self._log('dispatch', ticket)
            # Forward on the dispatcher thread: the next FIFO ticket cannot be
            # sent until this response has been read completely. The next-slot
            # deadline is measured from dispatch, so a slow response needs no
            # additional delay once it exceeds the interval.
            self._forward(ticket)
            self._queue.task_done()

    def _forward(self, ticket: Ticket) -> None:
        upstream_parts = urlsplit(self.upstream)
        request_parts = urlsplit(ticket.path)
        upstream_path = upstream_parts.path.rstrip('/')
        request_path = request_parts.path or '/'
        if upstream_path and (request_path == upstream_path or
                              request_path.startswith(upstream_path + '/')):
            suffix = request_path[len(upstream_path):] or '/'
            target = (f'{upstream_parts.scheme}://{upstream_parts.netloc}'
                      f'{upstream_path}{suffix}')
        else:
            target = self.upstream + ('/' if not request_path.startswith('/') else '') + request_path
        if request_parts.query:
            target += '?' + request_parts.query
        headers = {key: value for key, value in ticket.headers.items()
                   if key.lower() not in HOP_BY_HOP}
        request = Request(target, data=ticket.body if ticket.method not in ('GET', 'HEAD') else None,
                          headers=headers, method=ticket.method)
        response = None
        try:
            response = urlopen(request, timeout=self.request_timeout)
            ticket.status = response.status
            ticket.response_headers = [(key, value) for key, value in response.headers.items()
                                       if key.lower() not in HOP_BY_HOP]
            ticket.response_body = self._read_body(response)
        except HTTPError as error:
            ticket.status = error.code
            ticket.response_headers = [(key, value) for key, value in error.headers.items()
                                       if key.lower() not in HOP_BY_HOP]
            ticket.response_body = self._read_body(error)
        except (OSError, URLError, TimeoutError, ValueError) as error:
            ticket.error = f'{type(error).__name__}: {error}'
        except Exception as error:  # pragma: no cover - defensive boundary
            ticket.error = f'{type(error).__name__}: {error}'
        finally:
            if response is not None:
                response.close()
            with self._active_lock:
                self._active -= 1
            ticket.event.set()
            self._log('complete', ticket, status=ticket.status,
                      bytes=len(ticket.response_body), error=ticket.error)

    def _read_body(self, response) -> bytes:
        chunks = []
        size = 0
        while True:
            chunk = response.read(min(1024 * 1024, self.max_body_bytes - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > self.max_body_bytes:
                raise ValueError(f'upstream response exceeds {self.max_body_bytes} bytes')
            chunks.append(chunk)
        return b''.join(chunks)

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._dispatcher.join(timeout=2)


class GateHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    @property
    def gate(self) -> ApiGate:
        return self.server.api_gate  # type: ignore[attr-defined]

    def _serve(self) -> None:
        if self.path == '/healthz':
            payload = b'OK\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == '/gate/status':
            payload = json.dumps(self.gate.status(), separators=(',', ':')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self._error(400, 'invalid Content-Length')
            return
        if length < 0 or length > self.gate.max_body_bytes:
            self._error(413, 'request body is too large')
            return
        body = self.rfile.read(length) if length else b''
        headers = {key: value for key, value in self.headers.items()}
        ticket = self.gate.enqueue(self.command, self.path, headers, body)
        # The upstream timeout starts only after FIFO dispatch. Queue wait may
        # legitimately exceed it during a large batch, so wait for the ticket.
        ticket.event.wait()
        if ticket.error:
            self._error(502, ticket.error)
            return
        self.send_response(ticket.status)
        for key, value in ticket.response_headers:
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(ticket.response_body)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(ticket.response_body)

    def _error(self, status: int, message: str) -> None:
        payload = json.dumps({'error': message}, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        self._serve()

    def do_POST(self):  # noqa: N802
        self._serve()

    def do_PUT(self):  # noqa: N802
        self._serve()

    def do_DELETE(self):  # noqa: N802
        self._serve()

    def do_HEAD(self):  # noqa: N802
        self._serve()

    def log_message(self, format, *args):
        # Avoid logging Authorization headers; request metadata is in gate logs.
        return


def source_upstream(source_config: Path, provider: Optional[str]) -> str:
    source = tomllib.loads(source_config.read_text())
    name = provider or source.get('model_provider')
    providers = source.get('model_providers', {})
    try:
        value = providers[name]['base_url']
    except (KeyError, TypeError):
        raise ValueError(f'No base_url for provider {name!r} in {source_config}') from None
    if not isinstance(value, str):
        raise ValueError('source provider base_url must be a string')
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--upstream')
    parser.add_argument('--source-config', type=Path)
    parser.add_argument('--provider')
    parser.add_argument('--interval', type=float, default=10.0)
    parser.add_argument('--request-timeout', type=float, default=3600.0)
    parser.add_argument('--max-body-bytes', type=int, default=128 * 1024 * 1024)
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args(argv)
    if bool(args.upstream) == bool(args.source_config):
        parser.error('provide exactly one of --upstream or --source-config')
    upstream = args.upstream or source_upstream(args.source_config, args.provider)
    gate = ApiGate(upstream, interval=args.interval, request_timeout=args.request_timeout,
                   max_body_bytes=args.max_body_bytes)
    server = ThreadingHTTPServer((args.host, args.port), GateHandler)
    server.daemon_threads = True
    server.api_gate = gate  # type: ignore[attr-defined]
    address = f'http://{args.host}:{server.server_address[1]}'
    ready = {'url': address, 'host': args.host, 'port': server.server_address[1],
             'upstream': upstream, 'interval_seconds': args.interval}
    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps({'event': 'ready', **ready}, separators=(',', ':')), flush=True)
    def request_stop(signum, frame):
        # BaseServer.shutdown() waits for serve_forever() to return, so invoke
        # it off the signal-handler thread to avoid a self-deadlock.
        threading.Thread(target=server.shutdown, name='api-gate-shutdown',
                         daemon=True).start()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        gate.close()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
