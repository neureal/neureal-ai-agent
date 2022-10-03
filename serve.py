#!/usr/bin/python

import http.server
import socketserver
import os

PORT = 8080

os.chdir('/app/output')
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
