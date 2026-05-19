"""
Entry-point: запускает FastAPI (uvicorn) и Gradio в одном процессе.

Использование:
    python run.py                        # API на :8000, Gradio на :7860 (или следующем свободном)
    python run.py --api-only
    python run.py --gradio-only
    python run.py --api-port 8080 --gradio-port 7861
"""
import argparse
import io
import socket
import sys
import threading
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def is_port_free(port: int) -> bool:
    """Return True if the given TCP port is available on 0.0.0.0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def require_port(port: int, name: str = "Port") -> None:
    """Exit with a clear message if the port is already in use."""
    if not is_port_free(port):
        print(
            f"[ERROR] Port {port} is busy.\n"
            f"Close the program using port {port}, or specify another:\n"
            f"  python run.py --{name.lower()}-port <number>"
        )
        sys.exit(1)


def start_api(port: int = 8000):
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


def _start_gradio_direct(port: int = 7860):
    import gradio as gr
    from app.gradio_app import demo
    print(f"[*] Gradio UI: http://localhost:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1100px !important; }",
    )


def main():
    parser = argparse.ArgumentParser(description="Fashion AI launcher")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--gradio-only", action="store_true")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--gradio-port", type=int, default=7860)
    args = parser.parse_args()

    if args.api_only:
        require_port(args.api_port, "api")
        print(f"[*] FastAPI starting on http://0.0.0.0:{args.api_port} ...")
        start_api(args.api_port)
        return

    if args.gradio_only:
        require_port(args.gradio_port, "gradio")
        _start_gradio_direct(args.gradio_port)
        return

    require_port(args.api_port, "api")
    require_port(args.gradio_port, "gradio")

    import os
    os.environ["FASHION_API_URL"] = f"http://127.0.0.1:{args.api_port}"

    print(f"[*] FastAPI starting on http://0.0.0.0:{args.api_port} ...")
    api_thread = threading.Thread(target=start_api, args=(args.api_port,), daemon=True)
    api_thread.start()

    print("[*] Waiting for API to start (5s)...")
    time.sleep(5)

    _start_gradio_direct(args.gradio_port)


if __name__ == "__main__":
    main()
