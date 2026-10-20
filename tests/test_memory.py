import json
from pathlib import Path
from types import SimpleNamespace

from robodoctor.memory import EpisodeMemory, initial_todo, replace_section, write_plan_todo
from test_recovery import policy, ready, tool_call, completed, error_event, PAYLOAD_ERROR


def test_full_replacement_archives_changes_but_never_reads_notes(tmp_path, monkeypatch):
    todo = tmp_path/'TODO.md'
    notes = tmp_path/'NOTES.md'
    todo.write_text(replace_section(initial_todo(), 'TODO list', 'A: old plan'))
    notes.write_text('Detailed private-to-file step note\n')
    memory = EpisodeMemory(tmp_path)
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        assert path != notes, 'Long-term notes must never be automatically loaded'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    old = memory.capture(step=10)
    assert memory.capture(step=10) == old
    todo.write_text(replace_section(initial_todo(), 'TODO list', 'B: complete revised plan'))
    new = memory.capture(step=11)
    assert new['revision'] == 2
    assert 'B: complete revised plan' in new['content'] and 'A: old plan' not in new['content']
    assert 'Detailed private-to-file' not in json.dumps(new)
    archive = original_read(notes)
    assert archive.count('## TODO revision') == 2
    assert 'A: old plan' in archive and 'B: complete revised plan' in archive
    assert 'Detailed private-to-file step note' in archive


def test_episode_refresh_and_hybrid_plan_replace_preserve_current_geometry(tmp_path):
    todo = tmp_path/'TODO.md'
    text = replace_section(initial_todo(), 'Geometry and orientation', 'B wrist: wxyz [1,0,0,0] at step 5')
    text = replace_section(text, 'Task Progress', 'Holding object B; placement unconfirmed')
    todo.write_text(text)
    write_plan_todo(todo, [dict(id='A', status='ready')], 1, 4)
    write_plan_todo(todo, [dict(id='B', status='in_progress')], 2, 5)
    memory = EpisodeMemory(tmp_path)
    memory.episode(SimpleNamespace(task='pack', episode='ep-1', tick=5, phase='act', seed=2,
        obs=dict(instruction='Pack object B', remaining_steps=10), meta=dict(max_episode_steps=15),
        run=dict(reset_metadata=dict(eval_seed=0, layout_id=2, irrelevant_large_field='not included'))))
    content = memory.capture(step=5)['content']
    assert '"id":"A"' not in content and 'Current plan revision 1' not in content
    assert '"id":"B"' in content and 'Current plan revision 2' in content
    assert 'B wrist: wxyz [1,0,0,0] at step 5' in content
    assert 'Holding object B; placement unconfirmed' in content
    assert '"step_id": 5' in content and '"eval_seed": 0' in content
    assert 'irrelevant_large_field' not in content
    assert content.count('\n## ') == 4


def test_latest_todo_injected_in_service_result_and_413_recovery(policy, tmp_path):
    client = ready(tmp_path)
    memory = policy.memory
    memory.notes.write_text('LONG_TERM_SENTINEL: old failure narrative\n')
    memory.todo.write_text(replace_section(initial_todo(), 'TODO list', 'OLD_TODO_SENTINEL'))
    def edit_event(t):
        memory.todo.write_text(replace_section(initial_todo(), 'TODO list', 'NEW_TODO_SENTINEL'))
        return dict(method='item/completed', params=dict(threadId=t.thread, turnId=t.turn,
            item=dict(type='fileChange', id='edit-todo', status='completed')))
    policy.transport.events = [edit_event, error_event(), completed(PAYLOAD_ERROR),
                               tool_call(client, terminal=True), completed()]
    policy.run(client)
    turns = [p for method, p in policy.transport.requests if method == 'turn/start']
    assert 'OLD_TODO_SENTINEL' in turns[0]['input'][0]['text']
    recovery = turns[1]['input'][0]['text']
    assert 'NEW_TODO_SENTINEL' in recovery and 'OLD_TODO_SENTINEL' not in recovery
    result = json.loads(policy.transport.replies[-1][1]['contentItems'][0]['text'])
    assert 'NEW_TODO_SENTINEL' in result['todo_memory']['content']
    assert f'"step_id": {client.tick}' in result['todo_memory']['content']
    assert 'LONG_TERM_SENTINEL' not in json.dumps(turns + policy.transport.replies)
    archive = memory.notes.read_text()
    assert 'OLD_TODO_SENTINEL' in archive and 'NEW_TODO_SENTINEL' in archive
    assert f'## Step note 1 -> {client.tick}' in archive
    assert client.sim.calls.count('reset') == 1
