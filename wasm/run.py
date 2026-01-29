#!/usr/bin/env python3
from http import server


class RequestHandler(server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # These headers are required for js SharedArrayBuffer to work
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        server.SimpleHTTPRequestHandler.end_headers(self)


if __name__ == "__main__":
    server.test(HandlerClass=RequestHandler)
