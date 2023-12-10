# Voice-Conversion-Changer
AI voice use advanced algorithms to modify human voices, creating unique and creative effects. They're very popular in the entertainment industry, allowing for realistic character voices in movies, video games, streaming, and everyday communication.

# SoftVC VITS Singing Voice Conversion Fork

<p align="center">
  <a href="https://github.com/voicepaw/so-vits-svc-fork/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/voicepaw/so-vits-svc-fork/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://so-vits-svc-fork.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/so-vits-svc-fork.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/voicepaw/so-vits-svc-fork">
    <img src="https://img.shields.io/codecov/c/github/voicepaw/so-vits-svc-fork.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/so-vits-svc-fork/">
    <img src="https://img.shields.io/pypi/v/so-vits-svc-fork.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/so-vits-svc-fork.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/so-vits-svc-fork.svg?style=flat-square" alt="License">
</p>

A fork of [`so-vits-svc`](https://github.com/svc-develop-team/so-vits-svc) with **realtime support** and **greatly improved interface**. Based on branch `4.0` (v1) (or `4.1`) and the models are compatible.

## Features not available in the original repo

- **Realtime voice conversion** (enhanced in v1.1.0)
- Integrates [`QuickVC`](https://github.com/quickvc/QuickVC-VoiceConversion)
- Fixed misuse of [`ContentVec`](https://github.com/auspicious3000/contentvec) in the original repository.[^c]
- More accurate pitch estimation using [`CREPE`](https://github.com/marl/crepe/).
- GUI and unified CLI available
- ~2x faster training
- Ready to use just by installing with `pip`.
- Automatically download pretrained models. No need to install `fairseq`.
- Code completely formatted with black, isort, autoflake etc.

[^c]: [#206](https://github.com/voicepaw/so-vits-svc-fork/issues/206)

## Installation

### Option 1. One click easy installation

<a href="https://github.com/voicepaw/so-vits-svc-fork/releases/download/v1.3.2/install.bat" download>
  <img src="https://img.shields.io/badge/.bat-download-blue?style=flat-square&logo=windows" alt="Download .bat">
</a>

This BAT file will automatically perform the steps described below.

### Option 2. Manual installation (using pipx, experimental)

#### 1. Installing pipx

Windows (development version required due to [pypa/pipx#940](https://github.com/pypa/pipx/issues/940)):

```shell
py -3 -m pip install --user git+https://github.com/pypa/pipx.git
py -3 -m pipx ensurepath
```
Linux/MacOS:

```shell
python -m pip install --user pipx
python -m pipx ensurepath
```

#### 2. Installing so-vits-svc-fork

```shell
pipx install so-vits-svc-fork --python=3.10
pipx inject so-vits-svc-fork torch torchaudio --pip-args="--upgrade" --index-url=https://download.pytorch.org/whl/cu118 # https://download.pytorch.org/whl/nightly/cu121
```

### Option 3. Manual installation

<details>
  <summary>Creating a virtual environment</summary>

Windows:

```shell
py -3.10 -m venv venv
venv\Scripts\activate
```

Linux/MacOS:

```shell
python3.10 -m venv venv
source venv/bin/activate
```

Anaconda:

```shell
conda create -n so-vits-svc-fork python=3.10 pip
conda activate so-vits-svc-fork
```

Installing without creating a virtual environment may cause a `PermissionError` if Python is installed in Program Files, etc.

</details>

Install this via pip (or your favourite package manager that uses pip):

```shell
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118 # https://download.pytorch.org/whl/nightly/cu121
pip install -U so-vits-svc-fork
```

<details>
  <summary>Notes</summary>

- If no GPU is available or using MacOS, simply remove `pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118`. MPS is probably supported.
- If you are using an AMD GPU on Linux, replace `--index-url https://download.pytorch.org/whl/cu118` with `--index-url https://download.pytorch.org/whl/nightly/rocm5.6`. AMD GPUs are not supported on Windows ([#120](https://github.com/voicepaw/so-vits-svc-fork/issues/120)).
  </details>

### Update

Please update this package regularly to get the latest features and bug fixes.

```shell
pip install -U so-vits-svc-fork
# pipx upgrade so-vits-svc-fork
```

## Usage

### Inference

#### GUI

![GUI](https://raw.githubusercontent.com/voicepaw/so-vits-svc-fork/main/docs/_static/gui.png)

GUI launches with the following command:

```shell
svcg
```

#### CLI

- Realtime (from microphone)

```shell
svc vc
```

- File

```shell
svc infer source.wav
```

Pretrained models are available on [Hugging Face](https://huggingface.co/models?search=so-vits-svc) or [CIVITAI](https://civitai.com/?query=so-vits-svc).

#### Notes

- If using WSL, please note that WSL requires additional setup to handle audio and the GUI will not work without finding an audio device.
- In real-time inference, if there is noise on the inputs, the HuBERT model will react to those as well. Consider using realtime noise reduction applications such as [RTX Voice](https://www.nvidia.com/en-us/geforce/guides/nvidia-rtx-voice-setup-guide/) in this case.
- Models other than for 4.0v1 or this repository are not supported.
- GPU inference requires at least 4 GB of VRAM. If it does not work, try CPU inference as it is fast enough. [^r-inference]

[^r-inference]: [#469](https://github.com/voicepaw/so-vits-svc-fork/issues/469)

### Training

#### Before training

- If your dataset has BGM, please remove the BGM using software such as [Ultimate Vocal Remover](https://ultimatevocalremover.com/). `3_HP-Vocal-UVR.pth` or `UVR-MDX-NET Main` is recommended. [^1]
- If your dataset is a long audio file with a single speaker, use `svc pre-split` to split the dataset into multiple files (using `librosa`).
- If your dataset is a long audio file with multiple speakers, use `svc pre-sd` to split the dataset into multiple files (using `pyannote.audio`). Further manual classification may be necessary due to accuracy issues. If speakers speak with a variety of speech styles, set --min-speakers larger than the actual number of speakers. Due to unresolved dependencies, please install `pyannote.audio` manually: `pip install pyannote-audio`.
- To manually classify audio files, `svc pre-classify` is available. Up and down arrow keys can be used to change the playback speed.

[^1]: https://ytpmv.info/how-to-use-uvr/

#### Cloud

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voicepaw/so-vits-svc-fork/blob/main/notebooks/so-vits-svc-fork-4.0.ipynb)
[![Open In Paperspace](https://img.shields.io/badge/Open%20in-Paperspace-blue?style=flat-square&logo=paperspace)](https://console.paperspace.com/github/voicepaw/so-vits-svc-fork-paperspace/blob/main/so-vits-svc-fork-4.0-paperspace.ipynb)
[![Paperspace Referral](<https://img.shields.io/badge/Referral%20($10)-9VJN74I-blue?style=flat-square&logo=paperspace>)](https://www.paperspace.com/?r=9VJN74I)[^p]

If you do not have access to a GPU with more than 10 GB of VRAM, the free plan of Google Colab is recommended for light users and the Pro/Growth plan of Paperspace is recommended for heavy users. Conversely, if you have access to a high-end GPU, the use of cloud services is not recommended.

[^p]: If you register a referral code and then add a payment method, you may save about $5 on your first month's monthly billing. Note that both referral rewards are Paperspace credits and not cash. It was a tough decision but inserted because debugging and training the initial model requires a large amount of computing power and the developer is a student.

#### Local

Place your dataset like `dataset_raw/{speaker_id}/**/{wav_file}.{any_format}` (subfolders and non-ASCII filenames are acceptable) and run:

```shell
svc pre-resample
svc pre-config
svc pre-hubert
svc train -t
```

#### Notes

- Dataset audio duration per file should be <~ 10s.
- Need at least 4GB of VRAM. [^r-training]
- It is recommended to increase the `batch_size` as much as possible in `config.json` before the `train` command to match the VRAM capacity. Setting `batch_size` to `auto-{init_batch_size}-{max_n_trials}` (or simply `auto`) will automatically increase `batch_size` until OOM error occurs, but may not be useful in some cases.
- To use `CREPE`, replace `svc pre-hubert` with `svc pre-hubert -fm crepe`.
- To use `ContentVec` correctly, replace `svc pre-config` with `-t so-vits-svc-4.0v1`. Training may take slightly longer because some weights are reset due to reusing legacy initial generator weights.
- To use `MS-iSTFT Decoder`, replace `svc pre-config` with `svc pre-config -t quickvc`.
- Silence removal and volume normalization are automatically performed (as in the upstream repo) and are not required.
- If you have trained on a large, copyright-free dataset, consider releasing it as an initial model.
- For further details (e.g. parameters, etc.), you can see the [Wiki](https://github.com/voicepaw/so-vits-svc-fork/wiki) or [Discussions](https://github.com/voicepaw/so-vits-svc-fork/discussions).

[^r-training]: [#456](https://github.com/voicepaw/so-vits-svc-fork/issues/456)

### Further help

For more details, run `svc -h` or `svc <subcommand> -h`.

```shell
> svc -h
Usage: svc [OPTIONS] COMMAND [ARGS]...

  so-vits-svc allows any folder structure for training data.
  However, the following folder structure is recommended.
      When training: dataset_raw/{speaker_name}/**/{wav_name}.{any_format}
      When inference: configs/44k/config.json, logs/44k/G_XXXX.pth
  If the folder structure is followed, you DO NOT NEED TO SPECIFY model path, config path, etc.
  (The latest model will be automatically loaded.)
  To train a model, run pre-resample, pre-config, pre-hubert, train.
  To infer a model, run infer.

Options:
  -h, --help  Show this message and exit.

Commands:
  clean          Clean up files, only useful if you are using the default file structure
  infer          Inference
  onnx           Export model to onnx (currently not working)
  pre-classify   Classify multiple audio files into multiple files
  pre-config     Preprocessing part 2: config
  pre-hubert     Preprocessing part 3: hubert If the HuBERT model is not found, it will be...
  pre-resample   Preprocessing part 1: resample
  pre-sd         Speech diarization using pyannote.audio
  pre-split      Split audio files into multiple files
  train          Train model If D_0.pth or G_0.pth not found, automatically download from hub.
  train-cluster  Train k-means clustering
  vc             Realtime inference from microphone
```

#### External Links

[Video Tutorial](https://www.youtube.com/watch?v=tZn0lcGO5OQ)


