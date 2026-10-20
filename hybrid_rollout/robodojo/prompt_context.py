"""RSI context marker; no task-specific hints are injected."""
CONTEXT_VERSION = 'rsi_v1'

def task_context(task):
    return {}
