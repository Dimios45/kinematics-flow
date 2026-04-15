# Copyright (c) 2026 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class _State:
    GIT_PATH: str
    MODEL_CHECKPOINT: str
    DATA_PATH: str
    L_MAX: int
    MAX_DOF: int


_STATE: Optional[_State] = None


def _defaults() -> _State:
    current_file = os.path.abspath(__file__)

    def get_nth_parent(n: int) -> str:
        d = current_file
        for _ in range(n):
            d = os.path.dirname(d)
        return d

    git = get_nth_parent(3)
    return _State(
        GIT_PATH=git,
        MODEL_CHECKPOINT=os.path.join(git, "../checkpoint"),
        DATA_PATH=os.path.join(git, "../data"),
        L_MAX=2,
        MAX_DOF=22,
    )


def configure(cfg: Mapping[str, Any]) -> None:
    from omegaconf import DictConfig, OmegaConf

    global _STATE
    c = (
        OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, DictConfig)
        else dict(cfg)
    )
    gvars = c.get("globals", {}) or {}

    # fall back to sensible defaults if fields are missing
    dfl = _defaults()
    git = dfl.GIT_PATH

    _STATE = _State(
        GIT_PATH=git,
        MODEL_CHECKPOINT=os.path.join(git, "../checkpoint"),
        DATA_PATH=os.path.join(git, "../data"),
        L_MAX=int(dfl.L_MAX),
        MAX_DOF=int(gvars.get("MAX_DOF", dfl.MAX_DOF)),
    )


def __getattr__(name: str):
    if name in {
        "GIT_PATH",
        "MODEL_CHECKPOINT",
        "DATA_PATH",
        "L_MAX",
        "MAX_DOF",
    }:
        state = _STATE or _defaults()
        return getattr(state, name)
    raise AttributeError(name)
