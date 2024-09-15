# StructuralDPPIV

> This project is made possible by the [Wei Group](http://wei-group.net/) @ Shandong University. 
> All the code is under [MIT License](https://opensource.org/licenses/MIT).


Project code of the paper [StructuralDPPIV: A novel atom-structure based model for discovering dipeptidyl peptidase-IV inhibitory peptides.](https://academic.oup.com/bioinformatics/article/40/2/btae057/7596623)

## Installation


### Requirements

```text
dependencies:
  - Python 3.10.13 (recommonded)
  - torchaudio
  - torchvision
  - pytorch
  - cudatoolkit=11.3
  - pyg
  - rdkit
  - pytdc
  - pytorch-lightning==1.9
  - nb_conda
  - transformers
  - pynvml
  - grad-cam
  - tdqm
```



## Usage

### Training

For training the model, you can run `main/train.py` to train the model. The training process will be saved in the `./logs` directory. To see the training process, you can run `tensorboard --logdir ./logs` and open the browser to `http://localhost:6006/`.


### Testing

We provide a trained model in `https://drive.google.com/drive/folders/18Kb81AED8_dNEdQ_uQ8jYtyoPmOySd_a?usp=drive_link`. You can download the model and put it in the `./ckpt` directory. Then you can run `main/test.py` to test the model, or use your own data/trained model to see the results.




## Contacts

If you have any questions or suggestions, please contact us [here](http://wei-group.net/), or mail to scholarwd [at] gmail [dot] com.
