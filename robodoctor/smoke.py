"""Real RoboDojo tool smoke test with operator-selected pixels; no model call."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hybrid_rollout.robodojo.io import InputError, write_json
from .client import GroundedTools


def annotate_video(root, clicks, movement):
    """Overlay selected native pixels on recorded RGB; no generated imagery."""
    source = root/'sim/sensors.mp4'
    destination = root/'smoke_annotated.mp4'
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22) if Path(font_path).exists() else ImageFont.load_default()
    reader = imageio.get_reader(str(source))
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(str(destination), fps=fps, codec='libx264',
                               macro_block_size=2, output_params=['-movflags', '+faststart'])
    count = 0
    try:
        for index, frame in enumerate(reader):
            picture = Image.fromarray(frame)
            draw = ImageDraw.Draw(picture)
            chosen = next((c for c in reversed(clicks) if c['step_id'] <= index), clicks[0])
            # Original recorder has three 640x360 panels, head first.
            x = chosen['pixel'][0] * 640 / chosen['image_size'][0]
            y = chosen['pixel'][1] * 360 / chosen['image_size'][1]
            draw.ellipse((x-10, y-10, x+10, y+10), outline='#ff4040', width=3)
            draw.line((x-18, y, x+18, y), fill='#ff4040', width=2)
            draw.line((x, y-18, x, y+18), fill='#ff4040', width=2)
            draw.rectangle((0, 0, picture.width, 34), fill='#101820')
            label = f"Grounding pixel {tuple(chosen['pixel'])} | dual-arm control | step {index} | pre-operation approach only"
            draw.text((12, 4), label, font=font, fill='white')
            xyz = ', '.join(f'{v:.3f}' for v in chosen['position'])
            draw.rectangle((0, 320, picture.width, 360), fill='#101820')
            draw.text((12, 327), f"Surface in {chosen.get('frame', 'environment_origin')}: ({xyz}) m | red cross = selected head-image pixel", font=font, fill='#ffb0a0')
            rendered = np.asarray(picture)
            if index == 0:
                picture.save(root/'selected_pixel.png')
                for _ in range(int(fps)):
                    writer.append_data(rendered)
            writer.append_data(rendered)
            count += 1
        # Explicit presentation hold on final real frame, labelled as such.
        draw.rectangle((0, 0, picture.width, 34), fill='#101820')
        draw.text((12, 4), f"{movement['status'].upper()} | error {movement['position_error_m']*1000:.1f} mm, "
                  f"{movement['rotation_error_rad']:.4f} rad | final-frame hold; no grasp attempted", font=font, fill='#80ffb0')
        picture.save(root/'final_frame.png')
        for _ in range(int(2*fps)):
            writer.append_data(np.asarray(picture))
    finally:
        reader.close()
        writer.close()
    return dict(path=str(destination), source_frames=count, fps=fps, presentation_hold_seconds=3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='build_tower')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--port', type=int, default=19243)
    p.add_argument('--gpu', default='0')
    p.add_argument('--pixel', nargs=2, type=int)
    p.add_argument('--arm', choices=['left', 'right'], default='left')
    p.add_argument('--offset-world', nargs=3, type=float, default=[0, 0, .24])
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'sim/native_runtime').mkdir(parents=True)
    (root/'controller').mkdir()
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpu
    import sys
    command = [sys.executable, '-m', 'robodoctor.server', '--task', args.task,
               '--output', str(root/'sim'), '--port', str(args.port), '--eval-seed', str(args.eval_seed)]
    client = None
    clicks = []
    process = None
    try:
        with (root/'sim.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root/'sim/native_runtime', env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while '"event": "ready"' not in (root/'sim.log').read_text(errors='replace'):
            if process.poll() is not None:
                raise RuntimeError(f'Simulator exited ({process.returncode}); see {root}/sim.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Simulator startup exceeded 900 seconds')
            time.sleep(1)
        client = GroundedTools(root/'controller', args.task, sim_port=args.port, seed=args.seed, max_decisions=0)
        start = client.next_call(); start.pop('tool')
        packet = client.start(**start)
        write_json(root/'observation.json', packet)
        print(f'Observation ready: {root}/controller/observations/000/cam_high.png', flush=True)
        print(f'Write {root}/command.json with arm, pixel:[x,y], offset_world:[dx,dy,dz] or pose:[7 values].', flush=True)
        command_path = root/'command.json'
        if args.pixel:
            write_json(command_path, dict(arm=args.arm, pixel=args.pixel, offset_world=args.offset_world))
        previous = None
        deadline = time.monotonic() + 1800
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('No successful approach within 1800 seconds')
            if not command_path.exists() or command_path.read_text() == previous:
                time.sleep(.5)
                continue
            previous = command_path.read_text()
            command = json.loads(previous)
            arm = command['arm']
            try:
                grounded = client.grounding(client.request['request_id'], *command['pixel'])
                clicks.append(grounded)
                write_json(root/'grounding.json', grounded)
                print(json.dumps(grounded), flush=True)
                if 'pose' in command:
                    target = command['pose']
                else:
                    offset = np.asarray(grounded['world_axes_in_env']) @ np.asarray(command['offset_world'])
                    target = [*(np.asarray(grounded['position']) + offset).tolist(),
                              *packet['current_eef'][arm]['quaternion_wxyz']]
                targets = {name: dict(position=packet['current_eef'][name]['position'],
                    quaternion_wxyz=packet['current_eef'][name]['quaternion_wxyz'],
                    gripper_closed=bool(client.obs['states'][7*i+6] < .5))
                    for i, name in enumerate(('left', 'right'))}
                targets[arm].update(position=target[:3], quaternion_wxyz=target[3:])
                packet = client.act(observation_path=str(client.observation_path),
                    output_dir=client.next_call()['output_dir'], response=dict(
                        request_id=client.request['request_id'], mode='eef', target=targets,
                        steps=command.get('max_steps', 150), stop_on_reach=True,
                        reason=', '.join(f'{side}: ' + (
                            'coarse smoke approach to the grounded surface with scene-dependent clearance and gripper open'
                            if side == arm else 'hold measured pose and keep gripper open to stay clear')
                            for side in ('left', 'right'))))
                write_json(root/'observation.json', packet)
                write_json(root/'movement.json', packet['movement'])
                print(json.dumps(packet['movement']), flush=True)
                if packet['movement']['status'] == 'reached':
                    result = client.finish('smoke_approach_complete')
                    video = annotate_video(root, clicks, packet['movement'])
                    write_json(root/'smoke_result.json', dict(passed=True, movement=packet['movement'],
                        video=video, selected_points=clicks, native_result=result,
                        scope='Real simulator/tool integration; operator-selected pixel, no autonomous model or grasp test'))
                    print(f'Smoke passed. Video: {video["path"]}', flush=True)
                    break
                if client.phase == 'done':
                    raise RuntimeError('Native episode ended before approach reached')
                print('Approach not reached; inspect new observation and update command.json.', flush=True)
            except InputError as error:
                write_json(root/'input_error.json', dict(error=str(error), no_execution=True))
                print(f'Rejected before motion: {error}. Update command.json.', flush=True)
    finally:
        if client is not None:
            if client.phase != 'done' and client.episode is not None:
                client.finish('smoke_interrupted')
            # The annotated video is the smoke artifact; no hybrid report renderer.
            if client.sim is not None:
                client.sim.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
