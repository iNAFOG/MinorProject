"""Continue a saved, undecided Battleship pilot within its original total cap."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_harness.config import Config
from agent_harness.games.resume import resume_pilot, resume_evaluation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default='configs/battleship-gemini.yaml')
    parser.add_argument('--additional-requests',type=int,default=41)
    parser.add_argument('--request-interval',type=float,default=6)
    parser.add_argument('--evaluation-only',action='store_true',help='Finish an already selected, frozen evaluation')
    parser.add_argument('--request-timeout',type=float,default=90,help='HTTP transport timeout for frozen evaluation continuation')
    args=parser.parse_args()
    try:
        if args.evaluation_only:
            summary=resume_evaluation(Config.load(args.config),args.source,args.output,args.additional_requests,
                                      args.request_interval,args.request_timeout)
        else:
            summary=resume_pilot(Config.load(args.config),args.source,args.output,args.additional_requests,args.request_interval)
    except (ValueError,FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({key:summary[key] for key in ('status','stop_reason','request_attempts','total_tokens',
                                                 'accepted_changes','rejected_changes','results','continuation')},indent=2))
    raise SystemExit(0 if summary['status']=='complete' else 2)
