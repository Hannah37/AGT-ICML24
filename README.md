# AGT-ICML24
Official code repository of the paper Neurodegenerative Brain Network Classification via Adaptive Diffusion with Temporal Regularization (ICML 2024).

## Contribution 
1) Our method addresses the challenges of analyzing intricate brain networks by introducing a node-variant convolution that adaptively captures both localized homophily and heterophily characteristics.
2) Our method captures sequential variations in the progressive degeneration of brain networks, characterizing temporal features of a disease that change over time.
3) Consequently, AGT yields neuroscientifically interpretable results in both brain regional analysis and inter-group analysis.

## Datasets
All data required are available through [ADNI](https://adni.loni.usc.edu/) and [PPMI](https://www.ppmi-info.org/).
* To obtain cortical thickness from the ADNI study, T1-weighted MR images were parcellated into 160 brain regions based on the Destrieux atlas. From the MR images, skull stripping, tissue segmentation, and image registration were performed using Freesurfer.
* The region-wise average concentration level of FDG SUVR was calculated from PET scans based on the same Destrieux atlas, and the cerebellum was used as the reference region to calculate the SUVR.
* The preprocessed PPMI dataset can be accessed [here](https://github.com/brainnetuoa/data_driven_network_neuroscience?tab=readme-ov-file).

## Pretrained models
We provide the pretrained AGT models for all experiments (i.e., ADNI CT, ADNI FDG, and PPMI BOLD) in [this drive](https://drive.google.com/file/d/1aNH9fplja0_55GeN_E9sK4HWhkofbnQh/view?usp=sharing).
As described in the paper (Section 5.2), we used pseudo-labels of a pretrained Exact (Choi et al., MICCAI 2022) model at inference, and this Exact we used can be accessed in [this link](https://drive.google.com/file/d/1cUHMtGisUAizsebCaZzfPLfSUhX3-1Fw/view?usp=sharing).

## Running experiments
Before running AGT, unzip the ['trained_exact.zip'](https://drive.google.com/file/d/1cUHMtGisUAizsebCaZzfPLfSUhX3-1Fw/view?usp=sharing) in the root directory as shown below:
```
trained_exact/
models/
utils/
main.py
```

To train AGT from scratch and evaluate its performance with k-fold cross-validation, use the following commands:
```
python main.py --data=adni_ct --run_name=adni_ct
python main.py --data=adni_fdg --run_name=adni_fdg
python main.py --data=ppmi --run_name=ppmi --hidden_units=16
```

With the same main.py script, you can run other baseline models listed below.
* SVM
* MLP
* GCN (Kipf & Welling, ICLR 2017)
* GAT (Veličković et al., ICLR 2018)
* GDC (Gasteiger et al., NeurIPS 2019)
* GraphHeat (Xu et al., IJCAI 2019)
* ADC (Zhao et al., NeurIPS 2021)
* Exact (Choi et al., MICCAI 2022)

For example, you can run GraphHeat as follows:
```
python main.py --data=adni_ct --run_name=adni_ct_graphheat --model=graphheat
```

If you want to run other baseline models we used in our work not listed above (e.g., LSAP, BrainGNN, BrainNetTF), please contact Jayoon Sim (simjy98@postech.ac.kr).

For evaluation only, run 'train_test.py'. To run this, you first need to train models for k-folds and save all of them in the root directory. Otherwise, if you want to evaluate AGT, simply load [the pretrained AGT](https://drive.google.com/file/d/1aNH9fplja0_55GeN_E9sK4HWhkofbnQh/view?usp=sharing) and save it in the same root directory along with trained_exact.
```
python main_test.py --data=adni_ct --run_name=adni_ct_test
python main_test.py --data=adni_fdg --run_name=adni_fdg_test
python main_test.py --data=ppmi --run_name=ppmi_test --hidden_units=16
```

## Citation
If you would like to cite our paper, please use the BibTeX below.
```
@inproceedings{cho2024neurodegenerative,
  title={Neurodegenerative brain network classification via adaptive diffusion with temporal regularization},
  author={Cho, Hyuna and Sim, Jaeyoon and Wu, Guorong and Kim, Won Hwa},
  booktitle={Forty-first International Conference On Machine Learning},
  year={2024}
}
```




