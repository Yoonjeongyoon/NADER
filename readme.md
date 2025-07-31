# NADER: Neural Architecture Design via Multi-Agent Collaboration

This is official implementation for "NADER: Neural Architecture Design via Multi-Agent Collaboration" (CVPR2025)

## 📖 Overview

<p align="center">
  <img src='docs/nader.png' width=600>
</p>

## 🎉 News

July. 30, 2025: NADER is now open source.

## 🛠️ Prepare Environment and Datasets

### Installation the environment

Install [graphviz for python](https://github.com/xflr6/graphviz).

```bash
git clone git@github.com:yang-ze-kang/NADER.git
cd NADER
pip install requirements.txt
pip install -e .
```

### Download Datasets

## 💻️ Run

Set OpenAI API Key: Export your OpenAI API key as an environment variable. Replace "your_OpenAI_API_key" with your actual API key. Remember that this environment variable is session-specific, so you need to set it again if you open a new terminal session. On Unix/Linux:
```bash
export OPENAI_API_KEY="your_OpenAI_API_key"
export LLM_MODEL_NAME="your_OpenAI_name"
```
On Windows:
```bash
$env:OPENAI_API_KEY="your_OpenAI_API_key"
```

### [Optional] Craw papers and extract knowledge (Reader)
Your can use the inspiration extracted from CVPR2023 (data/paper/cvpr-2023) or run reader to extract knowledge by following the following steps or use the knowledge extracted by us at path 'database/inspirations/'
1. Craw paper
   ```bash
   python scripts/reader_craw_papers.py --cvpr-year 2023 --out-dir data/papers
   ```

2. Filter papers by abstracts
   ```bash
   python nader/scripts/reader_filter_papers_by_abstract.py --anno-path data/papers/cvpr-2023/annotations.json --abstract-dir data/papers/cvpr-2023/abstracts --anno-out-path data/papers/cvpr-2023/annotations_filted.json
   ```

3. Pdf2text
   ```bash
   python nader/scripts/reader_pdf2text.py --pdf-dir data/papers/cvpr-2023/papers --txt-dir data/papers/cvpr-2023/txts --token_num_path data/papers/cvpr-2023/txt_token_nums.txt
   ```

4. Extract knowledge
   ```bash
   # extract knowledge
   python nader/scripts/reader_extract_knowledge.py --anno-path data/papers/cvpr-2023/annotations_filted.json --txt-dir data/papers/cvpr-2023/txts --out-dir data/papers/cvpr-2023/txts_inspirations

   # merge all knowledge
   python nader/scripts/reader_merge_knowledge.py --inspirations-dir data/papers/cvpr-2023/txts_inspirations --out-path data/papers/cvpr-2023/inspirations.json
   ```

### Setting training script template
Setting your training script template in 'nader/train_utils/train_templates.py' to capable model training on the slurm.

### Run neural architecture design (Proposer & Modifier)
```bash
# Full pipeline (nas-bench-201)
python nader-nas-bench-201-full.py --max-iter 3 --dataset cifar10
```
### Reflect on experience (Reflector)
A example of track and reflect on experience is shown in 'reflector_extract_experience.py'.


## 📝 Citation
To cite NADER in publications, please use the following BibTeX entrie.
```bibtex
@InProceedings{Yang_2025_CVPR,
    author    = {Yang, Zekang and Zeng, Wang and Jin, Sheng and Qian, Chen and Luo, Ping and Liu, Wentao},
    title     = {NADER: Neural Architecture Design via Multi-Agent Collaboration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4452-4461}
}
```
