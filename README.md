# DigitalHumans Project

This repository contains the project-specific scripts for comparing **TokenHMR** and **4D-Humans** reconstruction results with **EgoHumans** ground-truth meshes.

The pipeline includes:

1. Preparing prediction outputs from TokenHMR or 4D-Humans.
2. Extracting EgoHumans head targets.
3. Running optimization.
4. Visualizing baseline and optimized meshes.
5. Computing evaluation metrics against EgoHumans ground truth.

Large data files, SMPL model files, prediction outputs, optimized meshes, and generated `.npz` / `.obj` files are **not included** in this GitHub repository. They should be placed locally according to the folder structure below.

---

## Folder structure

```text
Project/
├── README.md
├── .gitignore
│
├── smpl/
│   ├── README.md
│   └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
│
├── data/
│   ├── README.md
│   ├── colmap/
│   │   └── ...
│   │
│   ├── ego/
│   │   ├── aria01/
│   │   │   ├── calib/
│   │   │   ├── images/
│   │   │   └── undistort_map.npz
│   │   ├── aria02/
│   │   │   ├── calib/
│   │   │   ├── images/
│   │   │   └── undistort_map.npz
│   │   ├── aria03/
│   │   │   ├── calib/
│   │   │   ├── images/
│   │   │   └── undistort_map.npz
│   │   └── aria04/
│   │       ├── calib/
│   │       ├── images/
│   │       └── undistort_map.npz
│   │
│   ├── exo/
│   │   └── cam01/
│   │       └── undistorted_images/
│   │           ├── 00001.jpg
│   │           ├── 00002.jpg
│   │           ├── 00003.jpg
│   │           ├── 00004.jpg
│   │           └── ...
│   │
│   ├── mesh_cam/
│   │   └── cam01/
│   │       └── rgb/
│   │           ├── 00001/
│   │           │   ├── mesh_aria01.obj
│   │           │   ├── mesh_aria02.obj
│   │           │   ├── mesh_aria03.obj
│   │           │   └── mesh_aria04.obj
│   │           ├── 00002/
│   │           │   ├── mesh_aria01.obj
│   │           │   ├── mesh_aria02.obj
│   │           │   ├── mesh_aria03.obj
│   │           │   └── mesh_aria04.obj
│   │           └── ...
│   │
│   └── poses2d/
│       ├── aria01/
│       │   ├── left/
│       │   ├── rgb/
│       │   └── right/
│       ├── aria02/
│       ├── aria03/
│       ├── aria04/
│       └── cam01/
│           └── rgb/
│               ├── 00001.npy
│               ├── 00002.npy
│               └── ...
│
├── head_targets/
│   ├── README.md
│   ├── 00001_aria01_cam01_egohumans_style.txt
│   ├── 00001_aria02_cam01_egohumans_style.txt
│   ├── 00001_aria03_cam01_egohumans_style.txt
│   └── 00001_aria04_cam01_egohumans_style.txt
│
├── src/
│   ├── README.md
│   ├── baseline_mpjpe_same_regressor.py
│   ├── compute_total_loss.py
│   ├── make_aria_head_target.py
│   ├── smplify_v1_translation.py
│   ├── evaluate_optimized_vs_gt.py
│   ├── visualize_baseline_vs_egohumans_mesh3d.py
│   └── visualize_frame_before_after_all_people.py
│
├── TokenHMR/
│   └── demo_out/
│       ├── README.md
│       ├── my_image/
│       ├── my_image_smplify_adam/
│       ├── my_image_smplify_lbfgs/
│       └── my_image_smplify_v1/
│
├── 4D-Humans/
│   └── demo_out/
│       ├── README.md
│       ├── my_image/
│       ├── my_image_smplify_adam/
│       ├── my_image_smplify_lbfgs/
│       └── my_image_smplify_v1/
│
├── extract_all_aria_targets.sh
├── run_baseline_all.sh
├── run_smplify_frame.sh
├── visualize_baseline_vs_egohumans.sh
└── visualize_smplify_frame.sh
```

## Required local files

### 1. SMPL model

Place the SMPL model file (basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) here:

```text
smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```
This file is required because the scripts use the SMPL joint regressor to convert SMPL vertices into joints.

### 2. EgoHumans data

Place EgoHumans data under:
```text
data/
```
The local data/ folder should be copied from the shared course directory as follows:

```text
data/colmap   <- /work/courses/digital_human/team6/data/01_tagging/001_tagging/colmap
data/ego      <- /work/courses/digital_human/team6/data/01_tagging/001_tagging/ego
data/exo      <- /work/courses/digital_human/team6/data/01_tagging/001_tagging/processed_data/exo
data/mesh_cam <- /work/courses/digital_human/team6/data/01_tagging/001_tagging/processed_data/mesh_cam
data/poses2d  <- /work/courses/digital_human/team6/data/01_tagging/001_tagging/processed_data/poses2d
```

### 3. TokenHMR outputs
Copy the TokenHMR prediction outputs from:
```text
/work/courses/digital_human/team6/TokenHMR/demo_out/my_image
```
to:
```text
TokenHMR/demo_out/my_image/
```

Expected example files:
```text
TokenHMR/demo_out/my_image/00001_0.obj
TokenHMR/demo_out/my_image/00001_0_tkhmr.npz
```
After optimization, results are saved to folders such as:
```text
TokenHMR/demo_out/my_image_smplify_lbfgs/
TokenHMR/demo_out/my_image_smplify_adam/
```

### 4. 4D-Humans outputs
Copy the 4D-Humans prediction outputs from:
```text
/work/courses/digital_human/team6/4D-Humans/demo_out/my_image
```
to:
```text
4D-Humans/demo_out/my_image/
```

Expected example files:

```text
4D-Humans/demo_out/my_image/00001_0.obj
4D-Humans/demo_out/my_image/00001_0_4dhumans.npz
```
After optimization, results are saved to folders such as:
```text
4D-Humans/demo_out/my_image_smplify_lbfgs/
4D-Humans/demo_out/my_image_smplify_adam/
```

## Usage

All commands should be run from the project root:

```bash
cd Project
```
### 1. Run baseline evaluation

For TokenHMR:
```bash
./run_baseline_all.sh tokenhmr
```
For 4D-Humans:

```bash
./run_baseline_all.sh 4dhumans
```

This evaluates the original prediction meshes against EgoHumans ground-truth meshes.

### 2. Extract EgoHumans head targets

```bash
./extract_all_aria_targets.sh 00001 cam01
```

This creates files in:

```text
head_targets/
```

Example:
```text
head_targets/00001_aria03_cam01_egohumans_style.txt
```

These targets are used by the optimization step.

### 3. Run optimization (V1 - Translation Only)

For TokenHMR:
```text
./run_smplify_frame.sh 00001 tokenhmr adam "0:aria03,1:aria02,2:aria01,3:aria04"
```

For 4D-Humans:
```text
./run_smplify_frame.sh 00001 4dhumans adam "0:aria03,1:aria02,2:aria01,3:aria04"
```

Arguments:
```text
1st argument: frame ID, e.g. 00001
2nd argument: model name, tokenhmr or 4dhumans
3rd argument: optimizer, lbfgs or adam
4th argument: detection-to-Aria correspondence, e.g.,
              "0:aria03,1:aria02,2:aria01,3:aria04"
```

Optimized meshes are saved to:
```text
TokenHMR/demo_out/my_image_smplify_adam/
4D-Humans/demo_out/my_image_smplify_adam/
```

### 4. Run Optimization with 2d keypoints (V2 - SMPL Params jointly with global Rotation and Translation)
Apart from all the dependencies up to now, you need SMPL's torch layer
```pip install smplx```

Optionally, you can also run the following to setup the full conda environment required for this step if you are missing multiple dependencies
``` bash setup_conda_env.sh```

Run optimization through `run_smplify_kp2d_per_frame.sh`. An example invocation is below
```./run_smplify_kp2d_per_frame.sh 00001 tokenhmr adam "0:aria03,1:aria02,2:aria01,3:aria04" ```


### 5. Visualize baseline predictions

For TokenHMR:
```bash
./visualize_baseline_vs_egohumans.sh 00001 tokenhmr
```

For 4D-Humans:
```bash
./visualize_baseline_vs_egohumans.sh 00001 4dhumans
```

### 6. Visualize before and after optimization

For TokenHMR:
```bash
./visualize_smplify_frame.sh 00001 tokenhmr adam "0:aria03,1:aria02,2:aria01,3:aria04"
```
For 4D-Humans:
```bash
./visualize_smplify_frame.sh 00001 4dhumans adam "0:aria03,1:aria02,2:aria01,3:aria04"
```
### 7. Evaluate optimized meshes

Use the following command:

```bash
python src/evaluate_optimized_vs_gt.py \
  --frame 00001 \
  --model tokenhmr \
  --optimizer lbfgs \
  --smpl_model_dir ./smpl
```

Arguments:
```text
--frame:          frame ID, e.g. 00001
--model:          model name, tokenhmr or 4dhumans
--optimizer:      optimizer name, lbfgs or adam
--smpl_model_dir: SMPL model directory
```

The evaluation computes:
```text
Raw MPJPE
Root-aligned MPJPE
PA-MPJPE
Mean vertex error
Global root drift
```

The script performs automatic per-frame identity matching between detection IDs and EgoHumans identities.

Identity matching

Detection IDs are not guaranteed to stay consistent across frames.

For example, this mapping may be correct in one frame:
```text
det 0 -> aria03
det 1 -> aria02
```
but wrong in another frame.

Therefore, the evaluation scripts use automatic per-frame matching based on minimum root-aligned SMPL-joint MPJPE.

