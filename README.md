## Installation

This repository contains the code and data for the paper _Towards transferable metamodels for water distribution systems with
edge-based graph neural networks_ by  Bulat Kerimov, Riccardo Taormina, and Franz Tscheikner-Gratl. 

### Roadmap
- [x] EGNN Model
- [x] Pressure reconstruction
- [ ] Config files
- [x] Generated Data
- [x] Requirements


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/erytheis/egnn.git
   cd egnn
   ```
2. **Download the generated simulations**
   
   Download the foulder from [Google drive](https://drive.google.com/drive/folders/1VInz_m5JkcWan7le3SPiDziyisvI5PPa?usp=sharing) and save in the project directory inside ```saved```
   

3. **Set up a Python environment** (conda or venv recommended):
   ```bash
   conda create -n egnn_env python=3.8
   conda activate egnn_env
   ```

4. **Install required packages**:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install -r requirements.txt
   ```

## Running

