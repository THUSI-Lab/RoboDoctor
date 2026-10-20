"""The shared gate serializes actual upstream requests, not just dispatch starts."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from robodoctor.api_gate import ApiGate


@pytest.mark.parametrize('response_delay', [0.02, 0.18])
def test_fifo_waits_for_response_and_minimum_start_interval(response_delay):
    interval = 0.10
    seen = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal active, max_active
            length = int(self.headers['Content-Length'])
            value = self.rfile.read(length).decode()
            started = time.monotonic()
            with lock:
                active += 1
                max_active = max(max_active, active)
                seen.append([value, started, None])
                row = seen[-1]
            time.sleep(response_delay)
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'OK')
            self.wfile.flush()
            with lock:
                row[2] = time.monotonic()
                active -= 1

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    server_thread.start()
    gate = ApiGate(f'http://127.0.0.1:{upstream.server_address[1]}',
                   interval=interval, request_timeout=3)
    try:
        tickets = [gate.enqueue('POST', '/responses', {}, str(i).encode()) for i in range(3)]
        assert all(ticket.event.wait(3) for ticket in tickets)
        assert [ticket.status for ticket in tickets] == [200, 200, 200]
        assert [row[0] for row in seen] == ['0', '1', '2']
        assert max_active == 1
        for previous, current in zip(seen, seen[1:]):
            assert current[1] - previous[1] >= interval - 0.01
            assert current[1] >= previous[2] - 0.01
            if response_delay > interval:
                assert current[1] - previous[2] < 0.05
    finally:
        gate.close()
        upstream.shutdown()
        upstream.server_close()
        server_thread.join(timeout=2)
