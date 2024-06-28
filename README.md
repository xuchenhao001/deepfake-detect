# DeepFake Detection

DeepFake Detection Project

Preparation:

```bash
# convert video to frames
ffmpeg -i "./000_003.mp4" -qscale:v 2 -vf fps=1 "./manipulated/frame%05d.jpg"
```

Model training:

```bash
$ python3 main.py fit --config=config/nt_object.yaml
```

Model Testing:

```bash
$ python3 main.py test --config=config/nt_object.yaml --ckpt_path="./lightning_logs/version_0/checkpoints/epoch=0-step=38035.ckpt"
```

Tensorboard:

```bash
tensorboard --logdir=./lightning_logs/ --port=3414
```