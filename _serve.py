"""Minimal static file server for previewing the dashboard."""
import functools
import http.server
import os
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8755
directory = sys.argv[2] if len(sys.argv) > 2 else "output"
directory = os.path.abspath(directory)

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
print(f"serving {directory} at http://127.0.0.1:{port}")
httpd.serve_forever()
