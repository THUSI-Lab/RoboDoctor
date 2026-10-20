"""Small explicit Codex TOML adapter; isolated provider configuration."""
from __future__ import annotations
import os
import re
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import json


def read_config(path: str | os.PathLike) -> dict:
    value = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Codex config must be a TOML table")
    provider = value.get("model_provider")
    providers = value.get("model_providers", {})
    if not isinstance(provider, str) or not isinstance(providers, dict) or provider not in providers:
        raise ValueError("Codex TOML needs model_provider and [model_providers.<name>]")
    detail = providers[provider]
    if not isinstance(detail, dict) or not detail.get("base_url"):
        raise ValueError("Selected Codex provider needs base_url")
    return value


def prepare_codex(home: Path, config_path: Path | None, *, effort: str, model: str | None = None,
                  key_file: str | None = None, environ: dict | None = None,
                  source: dict | None = None, api_gate_url: str | None = None) -> tuple[dict, dict]:
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("invalid reasoning effort")
    source = read_config(config_path) if source is None else source
    provider_name = source["model_provider"]
    provider = source["model_providers"][provider_name]
    env = dict(os.environ if environ is None else environ)
    env_name = provider.get("env_key", "OPENAI_API_KEY")
    if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("Provider env_key must be an environment variable name, not a credential")
    credential = Path(key_file).read_text().strip() if key_file else env.get(env_name, "").strip()
    configured_bearer = provider.get("experimental_bearer_token")
    credential = credential or (configured_bearer or '').strip()
    if not credential:
        raise ValueError(f"No credential: set {env_name} or --key-file")
    provider = dict(provider)
    provider.pop('experimental_bearer_token', None)
    provider['env_key'] = env_name
    if api_gate_url:
        if not api_gate_url.startswith(('http://', 'https://')):
            raise ValueError('api_gate_url must be an absolute http(s) URL')
        provider['base_url'] = api_gate_url.rstrip('/')
    home.mkdir(parents=True, mode=0o700, exist_ok=False)
    config = dict(source)
    config["model"] = model or source.get("model", "gpt-6-astra")
    config["model_reasoning_effort"] = effort
    config["approvals_reviewer"] = source.get("approvals_reviewer", "auto_review")
    lines = [f'model = {json.dumps(config["model"])}',
             f'model_reasoning_effort = {json.dumps(effort)}',
             f'model_provider = {json.dumps(provider_name)}',
             'approvals_reviewer = "auto_review"']
    for key in ("wire_api", "model_reasoning_summary"):
        if key in source:
            lines.append(f"{key} = {json.dumps(source[key])}")
    lines.extend(["", f'[model_providers.{json.dumps(provider_name)}]'])
    for key, value in provider.items():
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"{key} = {json.dumps(value)}")
    (home/"config.toml").write_text("\n".join(lines) + "\n")
    (home/"config.toml").chmod(0o600)
    for name in ("CODEX_THREAD_ID", "CODEX_TURN_ID", "OPENAI_BASE_URL", "CODEX_API_KEY"):
        env.pop(name, None)
    env[env_name] = credential
    env.update(CODEX_HOME=str(home), ROLLOUT_CODEX_STATE_DIR=str(home),
               RSI_CODEX_PROVIDER=provider_name,
               OPENAI_API_KEY=credential, CODEX_IMAGE_MAX_EDGE="0")
    return env, dict(provider=provider_name, base_url=provider["base_url"], config=str(config_path))
