import yaml
from pathlib import Path
import os

def get_project_root():
    current_dir = Path(__file__).resolve().parent
    while not (current_dir / 'README.md').exists(): # Find dir with README.md (project root)
        if current_dir.parent == current_dir:
            return Path.cwd() 
        current_dir = current_dir.parent
    return current_dir

def load_config():
    # Load the configuration from config.yml.
    root = get_project_root()
    config_path = root / "src" / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if 'retrain_models' not in config:
        config['retrain_models'] = False

    config['root'] = root

    for key, value in config.items():
        if key.endswith('_dir') and value:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            config[key] = path
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    return config

settings = load_config()
