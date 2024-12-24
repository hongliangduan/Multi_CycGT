# Multi_CycGT: A DL-Based Multimodal Model for Membrane Permeability Prediction of Cyclic Peptides

This is the code for "Multi_CycGT: A DL-Based Multimodal Model for Membrane Permeability Prediction of Cyclic Peptides" paper.

## Directory Structure

```shell
bashCopy code/
├── data/            
├── data_process/  
├── model/         
├── LICENSE        
└── README.md       
```

## Quick-start

### Install dependency

Running this  command for installing dependency in docker:

```shell
pip install requirments.txt
./replace.sh
```

### Directory structure

Before start training the model, you need to process the dataset. The  `/data` contains datasets and preprocessing code.

### Training model

Running this  command for training the Multi_CycGT model：

```sh
python ./model/deep_learing/model_concat.py
```

### Membrane permeability prediction

Running this  command for predicting membrane permeability of cyclic peptides:

```sh
python ./model/deep_learing/model_concat.py
```

## Example

we provide a detailed example in the  notebook：[ Notebook](https://chat.openai.com/c/notebooks/Example.ipynb).

## Support or Report Issues

If you encounter any issues or need support while using Multi_CycGT, please report the issue in the [GitHub Issues](https://github.com/your_username/Multi_CycGT/issues) .

## Copyright and License

The project of Multi_CycGT follows [MIT License](https://chat.openai.com/c/LICENSE). Please read the license carefully before use.

## Update History

- v1.0.0 (2023-11-20):

   The first official version is released.

  - Supporting multi-mode data input.
  
  
  - Realizing the membrane permeability prediction.
  
  - Adding an example  notebook.

## Related Projects

The related projects about membrane permeability prediction and deep learning.

- [Related Project 1](https://github.com/related_project_1)
- [Related Project 2](https://github.com/related_project_2)

## Model Scalability

Running time of Multi_CycGT

![multi_cycgt](https://github.com/user-attachments/assets/0f1d02a7-655f-4b39-90a9-b3d0c314f5d5)

