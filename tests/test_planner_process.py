import os
import time
from types import SimpleNamespace

import pytest

from robodoctor import planner_process


def fake_worker(connection, config, log_path):
    connection.send(dict(ok=True, result=dict(ee_link='link6')))
    method, args, kwargs = connection.recv()
    if config['behavior'] == 'hang':
        time.sleep(30)
    elif config['behavior'] == 'crash':
        os._exit(1)
    else:
        connection.send(dict(ok=True, result=dict(status='Success', received=kwargs)))
        time.sleep(30)


@pytest.mark.parametrize('behavior', ['hang', 'crash'])
def test_native_hang_or_crash_is_bounded_without_replaying(monkeypatch, tmp_path, behavior):
    monkeypatch.setattr(planner_process, '_worker', fake_worker)
    worker = planner_process.PlannerProcess(dict(dt=.004, behavior=behavior),
        tmp_path/'planner.log', timeout=.1, startup_timeout=10)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='exceeded|exited'):
        worker.plan_joint([0]*6, [1]*6)
    assert time.monotonic()-started < 5
    assert not worker.process.is_alive()
    with pytest.raises(RuntimeError, match='unavailable'):
        worker.plan_joint([0]*6, [1]*6)
    worker.close()


def test_native_worker_preserves_ik_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(planner_process, '_worker', fake_worker)
    worker = planner_process.PlannerProcess(dict(dt=.004, behavior='success'),
        tmp_path/'planner.log', startup_timeout=10)
    try:
        assert worker.dt == .004 and worker.ee_link == 'link6'
        result = worker.solve_ik_to_joint(target_ee_pose=[0, 0, 1, 1, 0, 0, 0])
        assert result['received']['target_ee_pose'] == [0, 0, 1, 1, 0, 0, 0]
    finally:
        worker.close()


def test_install_shares_native_x5_config_and_preserves_support_arm(monkeypatch, tmp_path):
    calls = []
    def create(config, log):
        calls.append(config)
        return object()
    monkeypatch.setattr(planner_process, 'PlannerProcess', create)
    robot = SimpleNamespace(type='target', robot_type='arm', robot_name='x5',
        arm_joints_name=['joint1'], curobo_yml_path='x5.yml', entity_origin_pose=[0, 0, .765])
    support = SimpleNamespace(type='support')
    original = object()
    manager = SimpleNamespace(robot_list=[robot, robot, support], dt=.004,
        planner={'franka': original}, ik_solver={'franka': original})
    owned = []
    planner_process.install_target_planners(manager, tmp_path, owned)
    assert len(calls) == len(owned) == 1
    assert calls[0]['table_height'] == pytest.approx(-.025)
    assert manager.planner['x5'] is manager.ik_solver['x5'] is owned[0]
    assert manager.planner['franka'] is original
