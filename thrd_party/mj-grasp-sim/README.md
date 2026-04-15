# MuJoCo Grasping Simulator

The MuJoCo Grasping Simulator framework is presented in the paper https://arxiv.org/pdf/2410.18835.

## Installation Steps

### 1. Clone the Repository with Submodules

```bash
git clone https://github.com/boschresearch/mj-grasp-sim.git
cd mj-grasp-sim
```

### 2. Create and Activate Conda Environment

```bash
conda create -n mj-grasp-sim python=3.11 -y
conda activate mj-grasp-sim
```

### 3. Set Environment Variables

Replace `/path/to/your/in`, `/path/to/your/out` with your desired local directories.

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat <<EOF > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export MGS_INPUT_DIR=/path/to/your/in
export MGS_OUTPUT_DIR=/path/to/your/out
EOF
```

### 4. Create Required Directories

```bash
mkdir -p /path/to/your/in
mkdir -p /path/to/your/out
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 6. Download the Object Data

Download the [YCB dataset](https://bwsyncandshare.kit.edu/s/Ww4WES9AAqpRBMQ/download?path=%2F&files=YCB.zip) and place it under `/asset/mj-objects/`.
Then download the [GoogleScannedObjects (GSO) dataset](https://bwsyncandshare.kit.edu/s/Ww4WES9AAqpRBMQ/download?path=%2F&files=GoogleScannedObjects.zip) and place it under `/asset/mj-objects/`.
Next, run the `gso_to_delete.py` script on the GSO dataset to remove object
sets that behave unrealistically during simulation. For container usage, we
highly recommend excluding these asset files from the container image and
mounting them into the container at deployment time to keep the image smaller
and maintain flexibility.

## Usage

**mj-grasp-sim** provides several modules accessible via Hydra for flexible configuration.

### Run Modules with Hydra

- **Gripper Scan**

  ```bash
  python -m mgs.cli.scan_gripper <config>
  ```

- **Grasp Generation**

  ```bash
  python -m mgs.cli.gen_grasps <config>
  ```

- **Clutter Scene Generation**

  ```bash
  python -m mgs.cli.gen_clutter_scene <config>
  ```

- **Scene Rendering**

  ```bash
  python -m mgs.cli.render_scene <config>
  ```

*Replace `<config>` with the desired Hydra configuration
- All modules support Hydra for configuration management, allowing you to customize settings via configuration files or command-line overrides.
- Ensure that the specified input and output directories are correctly set in the environment variables.

## License

MuJoCo Grasping Simulator is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in MuJoCo Grasp Simulator, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
