"""One local command for a fresh end-to-end adaptation and transfer experiment."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_harness.cli.main import main


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/demo')
    parser.add_argument('--config',default=str(ROOT/'configs/e8.yaml'))
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a fresh output directory to reproduce from the initial harness')
    main(['benchmark','--config',args.config,'--dataset',str(ROOT/'benchmarks/coding.json'),
          '--state',str(output/'state.db'),'--output',str(output)])
