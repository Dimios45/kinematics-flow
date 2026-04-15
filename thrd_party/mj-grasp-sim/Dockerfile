# Copyright (c) 2025 Robert Bosch GmbH
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

FROM python:3.11-slim AS base-app

# Sets up main app directory
WORKDIR /app

RUN apt-get update && apt-get install -y locales libglew-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
	&& mkdir -p /out && mkdir -p /stats && mkdir -p /in

ENV LANG=en_US.utf8
ENV MUJOCO_GL=egl
ENV MGS_OUTPUT_DIR=/out
ENV MGS_INPUT_DIR=/in

# Copies the current directory contents into the container at /app
COPY . .
RUN pip install --upgrade pip && pip install -e .
# type in <script to execute>
ENTRYPOINT ["python", "-m"]

FROM base-app AS mgs-jax
RUN pip install -e .[jax]
ENTRYPOINT ["python", "-m"]

