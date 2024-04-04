# New York Congestion Tax Simulation
* Social Networking course final project
* model used: USTGCN
* project details: [presentation](https://docs.google.com/presentation/d/1nlA-2BOfXQxU7qbPJyIH-TnYKpbzSik3pXnChvElh4s/edit?usp=sharing)
[report](https://docs.google.com/document/d/1SUGsL3zP78imqTSDXKBv5IKQRvNO4whCKbR2qVjSMiM/edit?usp=sharing)


## USTGCN: Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network

![USTGCN](USTGCN.png?raw=true "Title")
Unified Spatio-Temporal Graph Convolutional Network, USTGCN. The unified spatio-temporal adjacency matrix, **A<sub>ST</sub>** showcases the cross space-time connections among nodes from different timestamps which consists of three types of submatrix: **A** as diagonal submatrix, **Ã** as lower submatrix and **0** as upper submatrix. **A<sub>ST</sub>**, a lower triangular matrix, facilitates traffic feature propagation from neighboring nodes only from the previous timestamps. The input features of different timestamps at convolution layer **l** are stacked into **X<sup>l</sup><sub>self</sub>** which is element-wise multiplied with broadcasted temporal weight parameter **W<sup>l</sup><sub>Temp</sub>** indicating the importance of the feature at the different timestamp. Afterwards, graph convolution is performed followed by weighted combination of self representation, **X<sup>l</sup><sub>self</sub>** and spatio-temporal aggregated vector,  **X<sup>l</sup><sub>ST</sub>**  to compute the representation **X<sup>l+1</sup><sub>self</sub>** that is used as input features at next layer, **l+1** or fed into the regression task.

### Model Architecture
![USTGCN Model](USTGCN_model.png?raw=true "Title")

To learn both daily and current-day traffic pattern, for each node we stack the traffic speeds of the last seven days  (traffic pattern during 09:30 AM - 10:30 AM for the last  week depicted with green color) along with the current-day traffic pattern for the past hour (traffic speed during 9:05 AM - 10:00 AM on current day i.e. Tuesrday depicted with red color) into the corresponding feature vector. We feed the feature matrix stacked for **N** nodes in the traffic network across **T = 12** timestamps to the USTGCN model of **K** convolution layers to compute spatio-temporal embedding. Finally, the regression module predicts future traffic intensities by utilizing the spatio-temporal embeddings.

### Comarison with Baselines
![Baseline Model](baseline_comparison.png?raw=true "Title")

### Envirnoment Set-Up 

Clone the git project:

```
$ git clone https://github.com/AmitRoy7781/USTGCN
```

Create a new conda Environment and install required packages (Commands are for ubuntu 16.04)

```
$ conda create -n TrafficEnv python=3.7
$ conda activate TrafficEnv
$ pip install -r requirements.txt
```

### Basic Usage:

Main Parameters:

```
--dataset           The input traffic dataset(default:PeMSD7)
--GNN_layers        Number of layers in GNN(default:3)
--num_timestamps    Number of timestamps in Historical and current model(default:12)
--pred_len          Traffic Prediction after how many timestamps(default:3)
--epochs            Number of epochs during training(default:200)
--seed              Random seed. (default: 42)
--cuda              Use GPU if declared
--save_model        Save model if declared
--trained_model     Run pretrained model if declaired
```

### Example Usage

**Train Model Using:**
```
$ python3 USTGCN.py --cuda --dataset PeMSD7 --pred_len 3 --save_model
```

<!-- **Run Trained Model:**

Please download the trained USTGCN models from [Google drive]() and place it in `saved_model/PeMSD7` folder

```
$ python3 USTGCN.py --cuda --dataset PeMSD7  --pred_len 3 --trained_model
```

**Run Trained Model:**

Please download the trained SSTGNN models from [Google drive]() and place them in `PeMSD7` folder

```
$ python3 USTGCN.py --cuda --dataset PeMSD7 --pred_len 3 --trained_model
```
!-->
 
### Cite

If you find our paper or repo useful then please cite our paper:

```bibtex
@inproceedings{roy2021unified,
  title={Unified spatio-temporal modeling for traffic forecasting using graph neural network},
  author={Roy, Amit and Roy, Kashob Kumar and Ali, Amin Ahsan and Amin, M Ashraful and Rahman, AKM Mahbubur},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

