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

from kin_flow.kin.gripper.allegro import (AllegroKinematicsModel,
                                          AllegroKinematicsModelNP)
from kin_flow.kin.gripper.dexee import (DexeeKinematicsModel,
                                        DexeeKinematicsModelNP)
from kin_flow.kin.gripper.panda import (PandaKinematicsModel,
                                        PandaKinematicsModelNP)
from kin_flow.kin.gripper.shadow import (ShadowKinematicsModel,
                                         ShadowKinematicsModelNP)
from kin_flow.kin.gripper.vx300 import (VX300KinematicsModel,
                                        VX300KinematicsModelNP)
from kin_flow.kin.gripper.z_0 import Z0KinematicsModel, Z0KinematicsModelNP

KINS = {
    "panda_orbit": lambda rngs: PandaKinematicsModel(is_static=True, rngs=rngs),
    "panda_static": lambda rngs: PandaKinematicsModel(is_static=True, rngs=rngs),
    "panda": lambda rngs: PandaKinematicsModel(is_static=False, rngs=rngs),
    "vx300": lambda rngs: VX300KinematicsModel(is_static=False, rngs=rngs),
    "dexee": lambda rngs: DexeeKinematicsModel(is_static=False, rngs=rngs),
    "dexee_static": lambda rngs: DexeeKinematicsModel(is_static=True, rngs=rngs),
    "allegro": lambda rngs: AllegroKinematicsModel(is_static=False, rngs=rngs),
    "allegro_real": lambda rngs: AllegroKinematicsModel(is_static=False, rngs=rngs),
    "shadow_static": lambda rngs: ShadowKinematicsModel(is_static=True, rngs=rngs),
    "shadow": lambda rngs: ShadowKinematicsModel(is_static=False, rngs=rngs),
    "z0": lambda rngs: Z0KinematicsModel(rngs=rngs),
}

KINS_NP = {
    "panda_orbit": PandaKinematicsModelNP(is_static=True),
    "panda_static": PandaKinematicsModelNP(is_static=True),
    "panda": PandaKinematicsModelNP(is_static=False),
    "vx300": VX300KinematicsModelNP(is_static=False),
    "dexee": DexeeKinematicsModelNP(is_static=False),
    "dexee_static": DexeeKinematicsModelNP(is_static=True),
    "allegro": AllegroKinematicsModelNP(is_static=False),
    "allegro_real": AllegroKinematicsModelNP(is_static=False),
    "shadow_static": ShadowKinematicsModelNP(is_static=True),
    "shadow": ShadowKinematicsModelNP(is_static=False),
    "z0": Z0KinematicsModelNP(),
}
