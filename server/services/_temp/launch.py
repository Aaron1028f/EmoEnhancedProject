# launch.py
import multiprocessing
import uvicorn

def run(service_module, port):
    uvicorn.run(f"{service_module}:app", host="127.0.0.1", port=port)

if __name__ == "__main__":
    services = [
        ("rag_service",    8001),
        ("llm_service",    8002),
        ("tts_service",    8003),
        ("video_service",  8004),
        ("orchestrator",   8000),
    ]
    procs = []
    for module, port in services:
        p = multiprocessing.Process(target=run, args=(module, port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
