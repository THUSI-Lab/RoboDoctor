"""Render mixed pi05/direct episodes without inventing absent policy proposals."""
import json
import math
import os
import traceback

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from hybrid_rollout.robodojo.io import write_json, sha256
from hybrid_rollout.robodojo.robodojo_server.debug_recorder import load_font
from hybrid_rollout.robodojo.robodojo_server.video_panel import fitted_lines, terminal_badge


def render_hybrid(recorder):
    output, controller = recorder.output, recorder.controller
    manifest = output/'manifest.json'
    if output.exists():
        value = json.loads(manifest.read_text()) if manifest.exists() else {}
        if value.get('status') == 'completed':
            return value
        raise FileExistsError(f'Refusing unfinished debug output: {output}')
    output.mkdir(parents=True)
    source = controller.parent/'sim/sensors.mp4'
    write_json(manifest, dict(status='rendering', pid=os.getpid(), source_video=str(source)))
    try:
        run = json.loads((controller/'run.json').read_text())
        records = json.loads((controller/'history.json').read_text())
        result_path = controller/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        segments, previous_end = [], 0
        for record in records:
            if not record['executed_steps']:
                continue
            if record['start_tick'] < previous_end or record['end_tick']-record['start_tick'] != record['executed_steps']:
                raise ValueError('Invalid hybrid history step mapping')
            segments.append(record)
            previous_end = record['end_tick']
        fps = 1/float(run['control_dt'])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid control rate')
        font, _ = load_font(22, recorder.font)
        temporary = output/'debug_rollout.partial.mp4'
        final = output/'debug_rollout.mp4'
        count, index = 0, 0
        with imageio.get_reader(str(source), input_params=['-threads', '2']) as reader:
            if not math.isclose(float(reader.get_meta_data()['fps']), fps, rel_tol=1e-4):
                raise ValueError('Sensor FPS differs from control rate')
            with imageio.get_writer(str(temporary), fps=fps, codec='libx264', macro_block_size=2,
                                   output_params=['-threads', '2', '-movflags', '+faststart']) as writer:
                for tick, frame in enumerate(reader):
                    rgb = Image.fromarray(frame[..., :3])
                    height = round(rgb.height*1280/rgb.width)//2*2
                    canvas = Image.new('RGB', (1280, height+320), '#111827')
                    canvas.paste(rgb.resize((1280, height)), (0, 0))
                    draw = ImageDraw.Draw(canvas)
                    action_tick = tick-1
                    while index < len(segments) and segments[index]['end_tick'] <= action_tick:
                        index += 1
                    row = segments[index] if index < len(segments) and segments[index]['start_tick'] <= action_tick else None
                    lines = [f'Hybrid | step {tick} | {tick/fps:.2f}s | {run.get("task", "")}']
                    if row:
                        policy = row['response']['mode'] == 'student'
                        lines.append(f'Source: {"pi05 reviewed prefix" if policy else "Agent direct control"} | '
                                     f'decision {row["decision"]} | plan revision {row.get("plan_revision", "?")}')
                        assessment = row.get('assessment') or (row.get('policy_review') or {}).get('assessment', {})
                        for arm in ('left', 'right'):
                            item = assessment.get(arm, {})
                            lines.append(f'{arm}: subtask {item.get("subtask_id", "direct/observe")} | '
                                         f'confidence {item.get("confidence", "n/a")}')
                        lines.extend(fitted_lines(row['response']['reason'], font, 1220, 4))
                    else:
                        lines.append('Initial observation' if tick == 0 else 'No matching execution receipt')
                    if result and tick >= result.get('step_id', math.inf):
                        lines.append(terminal_badge(result))
                    for i, line in enumerate(lines[:10]):
                        draw.text((24, height+12+i*29), line, font=font, fill='#f1f5f9')
                    writer.append_data(np.asarray(canvas))
                    count += 1
        if count == 0 or (result and count != result['step_id']+1):
            raise ValueError('Sensor video is incomplete for the recorded result')
        temporary.replace(final)
        write_json(output/'decisions.json', records)
        value = dict(status='completed', video_path=str(final), source_video=str(source),
            frames=count, fps=fps, duration_seconds=count/fps, playback_speed=1.0,
            reading_pause_frames=0, frame_mapping='frame 0 initial; frame t>0 action t-1 ACK',
            layout_version='hybrid_plan_v1', episode_result=result, original_video_unchanged=True,
            decisions_path=str(output/'decisions.json'), video_sha256=sha256(final))
        write_json(manifest, value)
        return value
    except Exception:
        write_json(manifest, dict(status='failed', error=traceback.format_exc(), source_video=str(source)))
        raise
