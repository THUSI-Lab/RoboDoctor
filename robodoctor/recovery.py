"""Model-request recovery without resetting or replaying simulator actions."""
import json
import re


def oversized_request(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    return bool(re.search(r'\b413\b', text) or any(marker in text for marker in (
        'payload too large', 'request body too large', 'request entity too large',
        'contextwindowexceeded', 'context_length_exceeded', 'context window exceeded',
        'maximum number of images', 'too many images', 'image limit exceeded')))


def closed_failed_turn(response, thread_id, turn_id):
    """Only a confirmed idle, failed turn permits a new model request."""
    thread = (response or {}).get('thread', {})
    if thread.get('id') != thread_id or thread.get('status', {}).get('type') != 'idle':
        return None
    turns = thread.get('turns', [])
    if not turns or turns[-1].get('id') != turn_id or turns[-1].get('status') != 'failed':
        return None
    return turns[-1]


def recovery_message(rollout, error, error_path, *, fresh_thread):
    # Bound provider text (e.g. HTML) and supply paths, never embed old transcripts,
    # images or action history in the recovery request itself.
    feedback = dict(error_type='model_request_failed',
        message=json.dumps(error, ensure_ascii=False)[:2000],
        full_error_path=str(error_path), request_cannot_be_sent=oversized_request(error),
        episode_id=getattr(rollout, 'episode', None), step_id=rollout.tick,
        phase=rollout.phase, next_call=rollout.next_call(),
        observation_path=str(rollout.observation_path) if getattr(rollout, 'observation_path', None) else None,
        request_id=getattr(rollout, 'request', {}).get('request_id'),
        fresh_model_thread=fresh_thread)
    text = ('The previous model request failed. This is error feedback, not a new simulation episode. '
        'The simulator and all completed actions are preserved. Do not reset, call robodojo_start '
        'again after it has succeeded, or repeat actions merely because a model response failed. '
        'Read the latest observation and recent relevant history from workspace.json paths; '
        'continue from the actual measured state and current next_call. Use only the current episode snapshot and current observation.\n')
    if getattr(rollout, 'method', None) != 'rsi_policy_doctor':
        text += ('Reconcile the injected TODO.md with current observations; query relevant '
                 'NOTES.md excerpts only if past evidence is needed.\n')
    if feedback['request_cannot_be_sent']:
        text += ('The previous request cannot be sent as constructed: its payload or context is too large. '
            'Do not repeat that request or reload the full conversation/history. A fresh model thread '
            'has been opened without the oversized history. Read only the current state and small '
            'relevant excerpts, view current images only as needed, and keep shell/file output small. '
            'Preserve native-image pixel coordinates for grounding. Change your request strategy '
            'and continue the task; this error does not mean a physical action is impossible.\n')
    return text + json.dumps(feedback, ensure_ascii=False)
