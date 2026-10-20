import json
import os

import imageio.v2 as imageio
import numpy as np
import pytest

from robodoctor.debug_video import RecordedDebugVideo, recover_sensors


def test_offline_retry_preserves_partial_and_renders_all_frames(tmp_path):
    controller = tmp_path/'controller'
    output = controller/'debug_video'
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(dict(status='interrupted')))
    (output/'debug_rollout.partial.mp4').write_bytes(b'previous partial file')
    (controller/'run.json').write_text(json.dumps(dict(control_dt=.04, evaluation_method='gpt_only')))
    (controller/'history.json').write_text('[]')
    (controller/'result.json').write_text(json.dumps(dict(complete=True, step_id=5)))
    (tmp_path/'sim').mkdir()
    with imageio.get_writer(str(tmp_path/'sim/sensors.mp4'), fps=25, codec='libx264') as writer:
        for tick in range(6):
            writer.append_data(np.full((32, 96, 3), tick * 40, dtype=np.uint8))
    recorder = RecordedDebugVideo(controller, retry_incomplete=True)
    result = recorder.render()
    assert result['status'] == 'completed'
    assert result['frames'] == 6 and result['duration_seconds'] == 6 / 25
    with imageio.get_reader(result['video_path']) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 6
    assert frames[-1][:32].mean() > frames[0][:32].mean() + 100
    archives = list(controller.glob('debug_video.incomplete_*'))
    assert len(archives) == 1
    assert (archives[0]/'debug_rollout.partial.mp4').read_bytes() == b'previous partial file'
    assert recorder.render() == result  # Completed output is never replaced.


def test_retry_rejects_live_legacy_renderer(tmp_path):
    output = tmp_path/'debug_video'
    output.mkdir()
    (output/'manifest.json').write_text(json.dumps(dict(status='rendering', pid=os.getpid())))
    with pytest.raises(RuntimeError, match='still alive'):
        RecordedDebugVideo(tmp_path, retry_incomplete=True).render()
    assert output.is_dir()
    assert not list(tmp_path.glob('debug_video.incomplete_*'))


def test_retry_rejects_concurrent_renderer(tmp_path):
    import fcntl
    with (tmp_path/'.debug_video.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already being rendered'):
            RecordedDebugVideo(tmp_path, retry_incomplete=True).render()


def test_recover_broken_mp4_from_recorded_rgb_preserves_original(tmp_path):
    controller = tmp_path/'controller'
    controller.mkdir()
    (controller/'run.json').write_text(json.dumps(dict(control_dt=.04)))
    (controller/'failure.json').write_text(json.dumps(dict(episode_id='episode', step_id=2)))
    observations = tmp_path/'sim/episode/observations'
    observations.mkdir(parents=True)
    source = tmp_path/'sim/sensors.mp4'
    source.write_bytes(b'broken MP4 without moov')
    for tick in range(3):
        frames = {key:np.full((32, 32, 3), tick*70, dtype=np.uint8)
                  for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
        np.savez(observations/f'{tick:06d}.npz', **frames)
    recover_sensors(controller)
    with imageio.get_reader(str(source)) as reader:
        assert reader.count_frames() == 3
        assert reader.get_data(2).mean() > reader.get_data(0).mean()+100
    assert next((tmp_path/'sim').glob('sensors.incomplete_*')).read_bytes() == b'broken MP4 without moov'
    original = source.read_bytes()
    recover_sensors(controller)
    assert source.read_bytes() == original
    source.write_bytes(b'broken again')
    (observations/'000001.npz').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        recover_sensors(controller)
    assert source.read_bytes() == b'broken again'


def test_sensor_recovery_rejects_live_simulator(tmp_path):
    (tmp_path/'processes.json').write_text(json.dumps(dict(sim_pid=os.getpid())))
    with pytest.raises(RuntimeError, match='still alive'):
        recover_sensors(tmp_path/'controller')


@pytest.mark.parametrize('finalized', [False, True])
def test_controller_only_renders_after_simulator_finalizes(tmp_path, finalized):
    from types import SimpleNamespace
    from robodoctor.client import GroundedTools
    calls = []
    client = GroundedTools.__new__(GroundedTools)
    client.output = tmp_path/'controller'
    client.sim = SimpleNamespace(close=lambda:calls.append('close'))
    client.recorder = SimpleNamespace(finalize=lambda:calls.append('render'))
    client.episode = 'episode'
    (tmp_path/'sim').mkdir()
    if finalized:
        (tmp_path/'sim/summary.json').write_text('{}')
    client.close()
    assert calls == (['close', 'render'] if finalized else ['close'])
