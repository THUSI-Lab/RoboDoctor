"""Render recorded debug video offline, without model calls or simulation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import uuid

from hybrid_rollout.robodojo.io import write_json
from hybrid_rollout.robodojo.robodojo_server.debug_recorder import DebugVideoRecorder


def recover_sensors(controller):
    """Offline rebuild from actual per-step RGB; never invent missing frames."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    controller = Path(controller)
    root = controller.parent
    processes = root/'processes.json'
    if processes.exists():
        pid = json.loads(processes.read_text()).get('sim_pid')
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Simulator PID {pid} is still alive; stop it before recovery')
    records = [root/'sim/summary.json', controller/'failure.json', controller/'result.json']
    evidence = next((json.loads(p.read_text()) for p in records
                     if p.exists() and json.loads(p.read_text()).get('episode_id')), None)
    if evidence is None:
        raise ValueError('No episode/step evidence for sensor recovery')
    directory = root/'sim'/evidence['episode_id']/'observations'
    last = evidence['step_id']
    paths = [directory/f'{tick:06d}.npz' for tick in range(last+1)]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('Incomplete per-step observations; refusing to fabricate sensor frames')
    run = json.loads((controller/'run.json').read_text())
    fps = 1/float(run['control_dt'])
    source = root/'sim/sensors.mp4'
    if source.is_file():
        try:
            with imageio.get_reader(str(source)) as reader:
                if reader.count_frames() == len(paths):
                    return  # Already complete and decodable.
        except (OSError, ValueError, RuntimeError):
            pass
    temporary = source.with_name('sensors.recovering.mp4')
    with imageio.get_writer(str(temporary), fps=fps, codec='libx264',
                            macro_block_size=2, output_params=['-threads', '2', '-movflags', '+faststart']) as writer:
        for path in paths:
            with np.load(path, allow_pickle=False) as obs:
                frames = [np.asarray(Image.fromarray(obs[key]).resize((640, 360)))
                          for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
                writer.append_data(np.concatenate(frames, axis=1))
    with imageio.get_reader(str(temporary)) as reader:
        if reader.count_frames() != len(paths):
            raise ValueError('Recovered sensor frame count mismatch')
    archive = None
    if source.exists():
        archive = source.with_name(f'sensors.incomplete_{uuid.uuid4().hex[:8]}.mp4')
        source.rename(archive)
    temporary.replace(source)
    write_json(root/'sim/sensors_recovery.json', dict(source_observations=str(directory),
        episode_id=evidence['episode_id'], frames=len(paths), fps=fps,
        preserved_video=str(archive) if archive else None, task_complete=False))


class RecordedDebugVideo(DebugVideoRecorder):
    def __init__(self, *args, retry_incomplete=False, **kwargs):
        kwargs.setdefault('encoder_threads', 2)
        super().__init__(*args, **kwargs)
        self.retry_incomplete = retry_incomplete

    def render(self):
        # Both normal finalization and offline recovery use this lock.
        with (self.controller/'.debug_video.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Debug video is already being rendered') from error
            if self.retry_incomplete and self.output.exists():
                manifest_path = self.output/'manifest.json'
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                if manifest.get('status') != 'completed':
                    # Legacy renderers did not acquire the lock. Do not move live output.
                    pid = manifest.get('pid')
                    if isinstance(pid, int) and pid > 0:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f'Previous renderer PID {pid} is still alive')
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    archive = self.output.with_name(f'{self.output.name}.incomplete_{stamp}_{uuid.uuid4().hex[:8]}')
                    self.output.rename(archive)
                    print(f'Preserved unfinished debug output: {archive}', flush=True)
            try:
                run = self.controller/'run.json'
                if run.exists() and json.loads(run.read_text()).get('control_version') == 'grounded_hybrid_v1':
                    from .hybrid_video import render_hybrid
                    return render_hybrid(self)
                return super().render()
            except KeyboardInterrupt:
                write_json(self.output/'manifest.json', dict(status='interrupted',
                    source_video=str(self.controller.parent/'sim/sensors.mp4'),
                    error='Rendering interrupted; use --retry-incomplete to rebuild offline'))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--retry-incomplete', action='store_true',
                        help='Archive unfinished debug output before rebuilding; never replace a live renderer')
    parser.add_argument('--recover-sensors', action='store_true',
                        help='Rebuild damaged sensor video from per-step observations after simulation exits')
    args = parser.parse_args()
    controller = args.run_root.resolve()/'controller'
    if not controller.is_dir():
        parser.error('Run must contain an existing controller directory')
    if args.recover_sensors:
        recover_sensors(controller)
    result = RecordedDebugVideo(controller, output=args.output,
                                retry_incomplete=args.retry_incomplete).finalize()
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
