from dataclasses import replace
import json
from agent_harness.config import condition
from agent_harness.core.budget import Budget
from agent_harness.core.trajectory import Trajectory
from agent_harness.models import MockModel,ModelResponse
from agent_harness.refinement.refiner import HarnessRefiner
from agent_harness.refinement.policy import validate_change
from agent_harness.storage.state import State


def test_fixed_model_can_propose_new_role_and_strategy(tmp_path):
    records=[{'task_id':task_id,'project':'p','family':'coding','verified':True,'error':None,'split':'adaptation','events':[]}
             for task_id in ('a','b')]
    payload={'changes':[{'change_type':'CREATE_AGENT','target_id':'dependency-reader','new_state':{
        'name':'Dependency reader','role':'dependency_analyzer','instructions':'Inspect dependency declarations and diagnose mismatches.',
        'capabilities':['coding']},'evidence':['a','b'],'reason':'Repeated dependency diagnosis'}]}
    model=MockModel(responses=[ModelResponse(content=json.dumps(payload),input_tokens=10,output_tokens=20)])
    cfg,state=condition(8),State()
    trajectory=Trajectory(tmp_path,'b',0,{})
    changes=HarnessRefiner().model_proposals(state,records,cfg,model,trajectory,Budget(cfg.limits))
    validate_change(changes[0],state,{r['task_id']:r for r in records},cfg)
    assert changes[0].new_state['role']=='dependency_analyzer'
    assert changes[0].new_state['status']=='candidate'
    assert not state.entries('agent') # proposing is not persistence
