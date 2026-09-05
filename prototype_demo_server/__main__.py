import argparse
import logging
import uvicorn
from .app import create_app
from .runtime import LocalRuntime, PROFILES, DEFAULT_MODEL

parser = argparse.ArgumentParser(description='Resident local VLM demo runtime; no model loaded until first request.')
parser.add_argument('--model', choices=list(PROFILES), default=DEFAULT_MODEL)
parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
parser.add_argument('--port', type=int, default=8010)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
uvicorn.run(create_app(LocalRuntime(args.model)), host=args.host, port=args.port, workers=1)
