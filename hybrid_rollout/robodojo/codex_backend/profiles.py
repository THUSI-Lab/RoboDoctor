"""Legacy launcher API profiles; explicit TOML configuration is preferred."""
import os


PROFILES = {
    'openai': ('api', 'openai', 'https://api.openai.com/v1'),
    'codex_a': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
}


def profile(name=None):
    name = name or os.environ.get('ROLLOUT_AUTH_PROFILE', 'openai')
    if name not in PROFILES:
        raise ValueError('Unknown rollout auth profile; no automatic fallback')
    mode, provider, url = PROFILES[name]
    return dict(name=name, auth_mode=mode, provider=provider, base_url=url)


def config_text(name=None):
    p = profile(name)
    text = (f'model_provider = "{p["provider"]}"\nmodel = "gpt-6-astra"\n'
            'model_reasoning_effort = "medium"\nmodel_reasoning_summary = "auto"\n')
    text += 'approvals_reviewer = "auto_review"\n'
    if p['auth_mode'] == 'chatgpt':
        text += 'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n'
    text += '\n[analytics]\nenabled = false\n'
    if p['auth_mode'] == 'api':
        text += (f'\n[model_providers.{p["provider"]}]\nname = "{p["provider"]}"\n'
                 f'base_url = "{p["base_url"]}"\nenv_key = "OPENAI_API_KEY"\nwire_api = "responses"\n')
    return text + '\n[features]\nfast_mode = false\ncodex_hooks = true\nhooks = true\n'
