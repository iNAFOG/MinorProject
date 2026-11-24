from dataclasses import replace
import sys
import pytest
from agent_harness.config import Limits
from agent_harness.environments.coding import CodingEnvironment
from test_core import fixture_task


def test_trusted_local_execution_and_bounded_output(tmp_path):
    task=fixture_task()
    disabled=CodingEnvironment(task,tmp_path/'disabled',Limits())
    with pytest.raises(PermissionError):
        disabled.execute([sys.executable,'-c','print("hello")'])
    enabled=CodingEnvironment(task,tmp_path/'enabled',Limits(max_output_chars=100),allow_local_execution=True)
    result=enabled.execute([sys.executable,'-c','print("x"*10000)'])
    assert result['returncode']==0 and len(result['output'])<=100


def test_trusted_command_check_without_docker(tmp_path):
    task=replace(fixture_task(),checks=({'kind':'command','argv':[sys.executable,'-c',
                  'from pathlib import Path; assert "def solve" in Path("solution.py").read_text()']},))
    env=CodingEnvironment(task,tmp_path/'workspace',Limits(),allow_local_execution=True)
    assert env.verify().success


def test_local_command_timeout(tmp_path):
    env=CodingEnvironment(fixture_task(),tmp_path/'workspace',Limits(max_execution_time=0.1),allow_local_execution=True)
    with pytest.raises(TimeoutError):
        env.execute([sys.executable,'-c','import time; time.sleep(10)'])
