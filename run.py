import argparse
import os
import socket
import sys
import threading
import time

def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False

def check_port(port: int, name: str):
    if not port_is_free(port):
        print(f"Порт {port} занят. Укажите другой: --{name}-port <номер>")
        sys.exit(1)

def run_api(port: int):
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")

def run_gradio(port: int):
    from app.gradio_app import demo
    print(f"Gradio UI: http://localhost:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--gradio-only", action="store_true")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--gradio-port", type=int, default=7860)
    args = parser.parse_args()

    if args.api_only:
        check_port(args.api_port, "api")
        run_api(args.api_port)
    elif args.gradio_only:
        check_port(args.gradio_port, "gradio")
        run_gradio(args.gradio_port)
    else:
        check_port(args.api_port, "api")
        check_port(args.gradio_port, "gradio")
        
        os.environ["FASHION_API_URL"] = f"http://127.0.0.1:{args.api_port}"
        
        print(f"Запуск API на порту {args.api_port}...")
        api_thread = threading.Thread(target=run_api, args=(args.api_port,), daemon=True)
        api_thread.start()
        
        print("Ожидание API (5 сек)...")
        time.sleep(5)
        
        run_gradio(args.gradio_port)