import argparse
import logging
import uvicorn
from .app import create_app

parser = argparse.ArgumentParser(description='Resident Gemma demo runtime; no model loaded until first request.')
parser.add_argument('--model', choices=['gemma3-4b'], default='gemma3-4b')
parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
parser.add_argument('--port', type=int, default=8010)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
