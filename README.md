<h1 align="center">Heterogeneous Graph Convolutional Neural Networks For Argument Pair Extraction</h1>

source code for paper: Heterogeneous Graph Convolutional Neural Networks For Argument Pair Extraction. Part of our code is based on [MLMC](https://github.com/ElevateSpirit/MLMC) and [MGF](https://github.com/HLT-HITSZ/MGF), thanks for their contributions.

## 1. Enviroments

- python: 3.8.16
- cuda: 11.3

## 2. Dependencies

- nltk: 3.8.1

- numpy: 1.22.3

- pandas: 2.0.1

- torch: 1.11.0+cu113
- termcolor: 2.3.0

- transformers: 4.21.1

- dgl-cu113: 0.9.1.post1

## 3. Dataset

[Review-Rebuttal](https://github.com/LiyingCheng95/ArgumentPairExtraction)

## 4. Usage

~~~sh
# step 1: put dataset in data folder
# step 2: create virtual envrioment
conda create -n HGAT python=3.8
# step 3: step into CrossModel directory, and install dependency
pip install -r requirment.txt
# step 4: train model
python main.py
~~~

## 5. Citation

## 6. Reference

~~~bibtext
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
@inproceedings{bao-etal-2021-argument,
    title = "Argument Pair Extraction with Mutual Guidance and Inter-sentence Relation Graph",
    author = "Bao, Jianzhu  and Liang, Bin and Sun, Jingyi and Zhang, Yice and Yang, Min and Xu, Ruifeng",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "3923--3934"

}
~~~

