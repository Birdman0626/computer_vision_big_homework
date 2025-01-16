# Computer-Vision project

This repo is the implementation of UI Reconition for AI3604 project.

## Repo Structure

```shell
.
├── Computer Vision Final Project.pdf #Requirements for project
├── Huggingface_agent 
│   ├── finetune # Code for finetuning
│   │   ├── finetune.py
│   │   ├── finetune_utils.py
│   │   └── loss_utils.py
│   ├── requirements.txt #requirements
│   ├── requirements_win.txt
│   ├── run.py #test file
│   ├── utils #origin utils file, deprecated
│   │   ├── crop.py
│   │   ├── icon_localization.py
│   │   ├── merge_strategy.py
│   │   ├── percept.py
│   │   └── text_localization.py
│   └── utils_fixed #our fixed utils for inference
│       ├── crop.py
│       ├── icon_localization.py
│       ├── merge_strategy.py
│       ├── percept.py
│       └── text_localization.py
├── README.md
├── ckpt #ckpt file for finetuning Grounding Dino, without origin model file
│   ├── checkpoint-2900
│   └── runs
├── clean.sh #shell scripts for finetuning
├── datasets #origin dataset, not cleaned
│   └── winmediaplayer...
├── eval #eval scripts given by TA
│   ├── ...
│   └── test_py
│       ├── main.py
│       └── ...
├── finetune.sh #shell scripts for finetuning
├── finetune_show.ipynb #visualization finetuning
├── inference.py #whole pipeline
├── label_scripts #tools for cleaning dataset
│   ├── dataset_dict.py
│   ├── label_analysis.py
│   ├── label_change_capital_scrollable.py
│   ├── rect_visual.py
│   └── xml_tensor_changes.py
├── model_classifier #The classifier model
│   ├── config #configs for model
│   │   ├── class_5_ablation.yaml
│   │   └── default.yaml
│   ├── dataset.pkl
│   ├── make_pkl.py
│   ├── model.py
│   └── train.py #training model
├── output_analysis #ROC curve for output in different scnarios
│   └── ...
├── output_visual #visualize our output
│   └── ...
├── requirements.txt 
├── requirements_all.txt #requirements for inference
└── run.sh #shell scripts for finetuning
```



## Environments

To build environments for inference pipeline:

```shell
pip install -r requirements_all.txt
```

To build environments for finetuning Grounding Dino: (choose only one command)

```shell
pip install -r Huggingface_agent/requirements.txt
pip install -r Huggingface_agent/requirements_win.txt
```

## usage

To run whole pipeline:

**Please first change the file path for model in inference.py**

```shell
python inference.py
```
To eval your outputs:

**Please first change all relative file path in eval/test_py/main.py**

```shell
cd eval/test_py
python main.py
```

To train Classifier model:

```shell
cd model_classifier
python make_pkl.py
python train.py
```

To finetune Grounding Dino model:

#TODO @YYM

To fix origin dataset and visualize:

```shell
python label_scripts/label_change_capital_scrollable.py
python label_scripts/label_analysis.py
python label_scripts/rect_visual.py
```

## Commit history

For huggingface-agent, download [model](https://jbox.sjtu.edu.cn/l/313Ker) and put files under model directory.

~~For My-agent, this will download model by `modelscope.snapdownload` under .cache file(in mac). Be aware of your disk space.~~

- My-agent has been removed. 2024.12.16 10:10

<!-- ## TODO

<img width="1093" alt="Screenshot 2024-12-10 at 12 05 31" src="https://github.com/user-attachments/assets/28eba408-6991-4211-956f-74271042234e"> -->

- 完成微调部分的代码。 2024.12.12 20:36
