#!/usr/bin/env python3
"""Launch the human annotation UI without importing any inference runtime."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--archive', type=Path, default=ROOT / 'TestData.zip')
    parser.add_argument('--annotations', type=Path, default=None)
    args = parser.parse_args()
    import uvicorn
    from physical_annotation.server import create_app
    app = create_app(ROOT, archive=args.archive, annotation_directory=args.annotations)
    print(f'Human annotation only: http://localhost:{args.port}')
    print('No labels are saved or verified until you review them in the browser.')
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
