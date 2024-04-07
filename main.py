import argparse
import os

from fastapi import FastAPI
from packages.routers import fm_router
from packages.runner import FastAPIRunner

app = FastAPI()

app.include_router(fm_router.fm)
# app.include_router(nfm)


@app.get('/')
def read_results():
    return {'msg': 'Main'}


if __name__ == "__main__":
    # python main.py --host 127.0.0.1 --port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    args = parser.parse_args()
    api = FastAPIRunner(args)
    api.run()