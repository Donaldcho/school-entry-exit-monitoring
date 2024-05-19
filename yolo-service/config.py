from dynaconf import Dynaconf

CONFIG = Dynaconf(
    settings_files=['settings.toml', '.secrets.toml'],
    environments=True,  # Enable environments handling
    env_switcher="ENV_FOR_DYNACONF",
    default_env="default"  # Force loading the default environment
)
