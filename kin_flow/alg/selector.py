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

from kin_flow.alg.flow.inference import inference as frame_inference
from kin_flow.alg.flow.input_target import input_target as frame_input_target
from kin_flow.alg.flow.loss import loss as frame_loss
from kin_flow.alg.flow.loss import stats as frame_stat


def get_inputs_targets(sample, cfg, key):
    if cfg.name == "Flow":
        inputs, targets = frame_input_target(sample, cfg, key)
    else:
        raise ValueError("Unknown algorithm")
    return inputs, targets


def inference(name, model, sample, num_samples, cfg):
    if name == "Flow":
        return frame_inference(model, sample, num_samples, cfg)
    else:
        raise ValueError("Unknown algorithm")


def get_train_fn(name):
    if name == "Flow":
        return frame_loss, frame_stat
    else:
        raise ValueError("Unknown algorithm")
