import argparse
import contextlib
import functools
import http.server
import os
import socket
import socketserver
import threading
import webbrowser

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "GET / " in (fmt % args) or ".html" in (fmt % args):
            print(f"  {fmt % args}")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def free_port(preferred: int | None) -> int:
    if preferred:
        return preferred
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    index = os.path.join(DOCS, "index.html")
    if not os.path.exists(index):
        print(f"docs/index.html not found at {index}")
        return 1
    assets = os.path.join(DOCS, "web-assets")
    missing = [n for n in ("text_enc.onnx", "image_enc.onnx", "port_cnn.onnx",
                           "vla.onnx", "bundle.json", "explorer.json")
               if not os.path.exists(os.path.join(assets, n))]
    if missing:
        print("missing browser assets: " + ", ".join(missing))
        print("regenerate them with: python scripts/export_web_model.py --run runs/main")
        return 1

    port = free_port(args.port)
    socketserver.TCPServer.allow_reuse_address = True
    try:
        srv = socketserver.TCPServer(
            ("127.0.0.1", port), functools.partial(Handler, directory=DOCS))
    except OSError as exc:
        print(f"could not bind port {port}: {exc}")
        return 1

    url = f"http://127.0.0.1:{port}/index.html"
    print(f"MMTG demo serving {DOCS}")
    print(f"  {url}")
    print("  press Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
