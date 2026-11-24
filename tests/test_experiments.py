from pathlib import Path
import json
import pytest
from agent_harness.config import condition,Config
from agent_harness.evaluation.benchmark import Dataset,Experiment
from agent_harness.evaluation.metrics import aggregate
from agent_harness.storage.state import State
from agent_harness.cli.main import main


def test_dataset_and_transfer_artifacts(tmp_path):
    dataset=Dataset('benchmarks/coding.json')
    state=State(tmp_path/'state.db')
    exp=Experiment(condition(8),state,tmp_path/'results',dataset.split('verification'))
    summary=exp.benchmark(dataset)
    assert summary['transfer_gain']==0 # mock already solves; do not invent capability gains
    assert summary['held_out_model_call_delta'] < 0
    assert summary['metrics']['accepted_changes'] > 0
    assert summary['metrics']['rejected_changes'] > 0
    assert summary['groups']['learned_held_out']['skill_reuse_rate']==1
    for name in ['experiment_summary.json','metrics.json','harness_history.json','comparison.csv','trajectories.jsonl']:
        assert (tmp_path/'results'/name).exists()
    frozen=state.fingerprint()
    for task in dataset.split('held_out'):
        exp.run_task(task)
    assert state.fingerprint()==frozen


def test_cli_and_yaml_configurations(tmp_path,capsys):
    for index in range(9):
        assert Config.load(f'configs/e{index}.yaml')==condition(index)
    main(['run','--task','benchmarks/example_task.json','--state',str(tmp_path/'state.db'),'--output',str(tmp_path/'run')])
    assert json.loads(capsys.readouterr().out)['verified']
    main(['inspect-harness','--state',str(tmp_path/'state.db')])
    assert json.loads(capsys.readouterr().out)['version']==0


def test_ablation_same_framework():
    cfg=condition(8).ablate('memory','skills','dynamic_agents','recursive_context','verification','self_refinement')
    assert not cfg.memory and not cfg.skills and not cfg.dynamic_agents
    assert not cfg.basic_verification and not cfg.verification_gate and not cfg.self_refinement


def test_cross_split_duplicate_rejected(tmp_path):
    raw=json.loads(Path('benchmarks/coding.json').read_text())
    duplicate=raw['tasks'][0] | {'task_id':'duplicate','split':'held_out'}
    raw['tasks'].append(duplicate)
    path=tmp_path/'leaked.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='leaked'):
        Dataset(path)
