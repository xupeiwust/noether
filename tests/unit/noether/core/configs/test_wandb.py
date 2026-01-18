#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from dataclasses import FrozenInstanceError

import pytest

from noether.core.configs.wandb import WandBConfig, WandBMode


def test_wandb_mode_value_and_properties() -> None:
    # Ensure value property and mode checks:
    assert isinstance(WandBMode.ONLINE.value, str)
    assert WandBMode.ONLINE.value == "online"
    assert WandBMode.OFFLINE.value == "offline"
    assert WandBMode.DISABLED.value == "disabled"

    # Identity comparison via is:
    assert WandBMode.ONLINE is WandBMode("online")
    assert WandBMode.OFFLINE is WandBMode("offline")


def test_wandb_config_valid_online() -> None:
    # Valid configuration in ONLINE mode:
    cfg = WandBConfig(
        mode=WandBMode.ONLINE,
        host="https://wandb.ai",
        entity="user123",
        project="my-project",
    )
    assert cfg.is_online
    assert not cfg.is_offline
    assert not cfg.is_disabled


def test_wandb_config_missing_fields_in_online() -> None:
    # Missing host should raise TypeError:
    with pytest.raises(TypeError):
        _ = WandBConfig(mode=WandBMode.ONLINE, host=None, entity="e", project="p")
    # Missing entity:
    with pytest.raises(TypeError):
        _ = WandBConfig(mode=WandBMode.ONLINE, host="h", entity=None, project="p")
    # Missing project:
    with pytest.raises(TypeError):
        _ = WandBConfig(mode=WandBMode.ONLINE, host="h", entity="e", project=None)


def test_wandb_config_disabled_allows_none() -> None:
    # Disabled mode allows None for host/entity/project:
    cfg = WandBConfig(mode=WandBMode.DISABLED)
    assert cfg.is_disabled
    assert not cfg.is_online
    assert not cfg.is_offline


def test_require_online_raises() -> None:
    # OFFLINE mode should raise:
    cfg_off = WandBConfig(mode=WandBMode.OFFLINE, host=None, entity=None, project=None)
    with pytest.raises(RuntimeError):
        cfg_off.require_online()
    # DISABLED mode should also raise:
    cfg_dis = WandBConfig(mode=WandBMode.DISABLED, host=None, entity=None, project=None)
    with pytest.raises(RuntimeError):
        cfg_dis.require_online()
    # ONLINE mode should not raise:
    cfg_on = WandBConfig(mode=WandBMode.ONLINE, host="h", entity="e", project="p")
    cfg_on.require_online()


def test_wandb_config_immutable() -> None:
    cfg = WandBConfig(mode=WandBMode.ONLINE, host="h", entity="e", project="p")
    with pytest.raises(FrozenInstanceError):
        cfg.host = "other"  # Note: IDE will highlight it in red but this is expected! We *don't* want edit after init.
