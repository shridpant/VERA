# VERA research fork

This is [`shridpant/VERA`](https://github.com/shridpant/VERA), a fork of the official VERA code,
[`vera-framework/VERA`](https://github.com/vera-framework/VERA) (Ye, Liu & He, *Explainable Video Anomaly
Detection via Verbalized Learning of Vision-Language Models*, CVPR 2025). The upstream code and history are kept
unchanged up to commit `15b8bcb8`. The upstream README is reproduced verbatim [at the end of this
file](#upstream-vera-readme). Please cite the VERA paper when you use this code.

The fork adds a reproducible research setup around the upstream code:

- **One source change.** It makes `generate_initial_score.py` importable for one-video inference. It is kept in
  its original commit, `008ed401`, directly on top of `15b8bcb8`. See [`setup/patches/`](setup/patches) and
  [`setup/source/README.md`](setup/source/README.md).
- **`setup/`.** The environment specification, runners, validators, tests and setup logs.
- **`shared/`.** Provenance and checksums for the external model, videos, frames, features and reference scores.
  The data itself is not redistributed; see [`shared/README.md`](shared/README.md).
- **`experiments/`.** Executed experiments, registered in [`EXPERIMENTS.md`](EXPERIMENTS.md). Their configs,
  manifests, compact raw records and reports were imported after the runs, unchanged.

## Research setup

The setup lives on RIT's Oz GPU host as `~/vera_research`, which is the working tree of this fork. It is for
studying VERA. VERA has a frozen VLM (InternVL2-8B) answer learned guiding questions about 8-frame windows, then
turns the 0/1 answers into frame-level anomaly scores in three steps:

1. **VLM decisions:** a 0/1 label and a short explanation per window.
2. **Scene-context refinement:** ImageBind similarity between segments.
3. **Temporal refinement:** Gaussian smoothing plus a position weighting.

The longer-term question is whether VERA-style explanations help people judge alerts. Nothing here answers that yet.

Status: see [`setup/SETUP_STATUS.md`](setup/SETUP_STATUS.md). Experiments: see [`EXPERIMENTS.md`](EXPERIMENTS.md).

### Layout

```text
(root)      upstream VERA code (Data/, Inference/, Preprocessing/, training.py, generate_initial_score.py, ...)
setup/      connection.md, SETUP_STATUS.md
            source/README.md   which source the runs import; source/VERA itself is a local checkout (not tracked)
            patches/           the local VERA change (commit 008ed401) as a git format-patch
            environment/       requirements, pip_freeze.txt, conda_explicit.txt, REBUILD.md (env/ is local only)
            scripts/           launch.sh, run_experiment.py, validate_run.py, summarize_run.py, tests, fetchers
            logs/              setup logs, smoke tests, archive listings
shared/     README.md + provenance only; model, videos, frames, features and reference outputs are local
experiments/experimentNN/      README.md, config.yaml, runs/run_NNN/ (never overwritten)
```

### Adding an experiment

1. Create `experiments/experimentNN/` with a `README.md` (purpose) and a `config.yaml`. Copy experiment01's config
   and change the video, prompt or sampling fields.
2. Add a row to `EXPERIMENTS.md`.
3. Run through the launcher. It pins one idle ≥40 GB GPU by UUID and does not use a scheduler:

   ```bash
   RUN=experiments/experimentNN/runs/run_001
   bash setup/scripts/launch.sh --name expNN --log-dir $RUN/logs -- \
     setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experimentNN/config.yaml --run-dir $RUN --new-run
   setup/environment/env/bin/python setup/scripts/validate_run.py --run-dir $RUN
   ```

The config's `source` block names the checkout the runner imports (`checkout`) and the commit it must be at
(`expected_head`). The runner refuses a dirty checkout or a head that differs from the config. Old runs keep
pointing at the revision they used. Shared caches are reused only when checksums match. The frame cache checks the
source video sha256, the extractor sha256, the cv2 version and every frame's sha256.

Videos without released VERA features would need the LAVAD/ImageBind feature runner. It is **not set up yet**; see
`setup/environment/requirements-imagebind-optional.txt` and `setup/SETUP_STATUS.md`.

### Licence and attribution

- **Upstream code.** It has no licence file at `15b8bcb8`, and the code remains the VERA authors' work. This fork
  adds no licence of its own.
- **External artifacts.** These keep their own terms and are not redistributed here: InternVL2-8B (OpenGVLab;
  model card: MIT), the UCF-Crime videos and annotations (UCF CRCV), and the VERA authors' released features and
  initial scores (the Google Drive links in the upstream README).

---

<a id="upstream-vera-readme"></a>
**Upstream VERA README.** Reproduced verbatim from `vera-framework/VERA@15b8bcb8`.

# VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models
This is the code repo for the paper [`VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models`](https://vera-framework.github.io/) (CVPR 2025).

Our paper presents a video anomaly detection (VAD) method utilizing vision-language models (VLMs) without the need of using extra reasoning modules or instruction tuning datasets: **we just need to input the prompts found by verbalized learning with a given VLM to get VAD results.** This style will eliminate the construction of fine-tuning dataset or the introduction of additional LLMs like ChatGPT. 

**What can this repo do for us?**
With this repo, we can:
* Use the training codes to find the best prompt for your deployed VLMs in video anomaly detection (or other vision tasks)
* Use the inference codes and provided prompts/initial scores to run our initial score generation/smoothing codes

**Note**

In this repository, we use InternVL2 as an example to illustrate the use of the provided open-source code.

## 0. Environment and Data


The construction of environment for running VERA relies on the backbone we choose. Please install the environment based on the backbone we want to run VERA on. For example, in this repo, we use InternVL2, we will install the environment based on the instruction provided by InternVL2 (https://internvl.readthedocs.io/en/latest/get_started/installation.html). We also install the PyTorch Lighting (https://lightning.ai/docs/pytorch/stable/) package.

For datasets, we run on UCF-Crime and XD-Violence. Please download the original videos from links provided by the authors. UCF-Crime's project page is https://www.crcv.ucf.edu/projects/real-world/ (we recommend using the dropbox link) and XD-Violence's project page is https://roc-ng.github.io/XD-Violence/. Please let us know if you cannot download from the official links and we are happy to help. 

With the downloaded videos, we need to extract frames from the original videos. Please use the provided code in the `Preprocessing` folder for getting extracted frames. After that, we can put the frames in the `Data` folder

## 1. Training in VERA
We can run the following script to obtain the optimal guiding question. The default batch size is 2 and the number of sampled frames is 8. After training, we usually choose the one with the highest validation accuracy as the guiding questions used in testing (we can also use AUC as the metric for selecting the guiding questions).
```
python training.py
```
**Note: If you would like to train on XD-Violence, please uncomment L303-L305 in training.py. This will load the XD-Violence training and validation data.**


For UCF-Crime, the best guiding questions we find are as follows:
```
1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?
2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?
3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?
4. Is there any visible damage or unusual movement in the video that indicates an anomaly?
5. Are there any unusual sounds or noises in the video that suggest an anomaly?
```

For XD-Violence, the best guiding questions we find are as follows:
```
1. Are there any frames where the video content shows an object or person performing an unusual action that does not fit the usual pattern of the video content?
2. Are there any frames where the video content shows an object or person behaving in a way that is clearly abnormal or unexpected?
3. Are there any frames where the video content shows a clear indication of an anomaly in the behavior of objects or people, such as an object moving in an unexpected way or a person performing an unusual action?
4. Are there any frames where the video content shows a player or object being out of place or not fitting the usual pattern of the video content?
5. Are there any frames where the video content shows a disruption in the normal flow of the video, such as a sudden change in the camera angle or a loss of video quality?
```

## 2. Inference in VERA 
After finishing training and deciding the guiding questions we will use, we need to do inference for test videos. The related codes are put in the `Inference` folder. We can put the learned questions in the backbone and run them on test data to get the initial anomaly score in the segment level. This is Step 1 in VERA's testing (Please refer to Sec. 3.3. in our paper). For this step, we need to run this script after putting guiding questions in the variable ```new_qs``` in the script.

```
python generate_initial_score.py
```

**Note**: We have provided the initial scores obtained by VERA (backbone: InternVL2) on UCF-Crime with this [link](https://drive.google.com/file/d/1LIvPUE7Pv7tPxtXjpOU0sOpSZs6Y7xGv/view?usp=drive_link). We have also provided the initial scores obtained by VERA (backbone: InternVL2) on XD-Violence with this [link](https://drive.google.com/file/d/113rhslKnIE7CKeWd0HKiy8FWp0QJ5OxH/view?usp=drive_link).

Next, given the initial score, we need to run Step 2 and Step 3 introduced in Sec. 3.3 for getting the frame-level anomaly score. We put those two steps in one script and we just need to run this line:

```
python compute_auc.py
```

Please note that the inference scores requires us to extract the vision feature beforehand. We use LAVAD's (https://github.com/lucazanella/lavad) and provide code and use ImageBind for extracting the vision features of each segment. You can access the extracted features we use for UCF-Crime from this [link](https://drive.google.com/file/d/1gwkgIKiPg6fpRaO5yXkxuRcN3X1-97LF/view?usp=drive_link) directly. The extracted feature for XD-Violence is provide in this [link](https://drive.google.com/file/d/1_bN5TRYPpaFpLpf_M7YniRwWeOZgNJtI/view?usp=sharing).

**Note**: We have provide the computation of AUC and AP for XD-Violence dataset as well, which is in the code ```compute_auc_XD_Violence.py``` and ```compute_AP_XD_Violence.py```, respectively. For AP computation in XD-Violence, we follow [Wu et al.](https://roc-ng.github.io/XD-Violence/) for the preprocessing and postprocessing of getting ground truth and smoothing. 

## Acknolwedgement
We thank [Zanella et al.](https://github.com/lucazanella/lavad), [Lv et al.](https://github.com/ktr-hubrt/VAD-LLaMA), and [Wu et al.](https://roc-ng.github.io/XD-Violence/) for sharing their codes.


## Citation
If you use this code or find our work helpful, please consider citing our paper. Thank you!
```
@InProceedings{Ye_2025_CVPR,
    author    = {Ye, Muchao and Liu, Weiyang and He, Pan},
    title     = {VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {8679-8688}
}
```
