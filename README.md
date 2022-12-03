# bi-est
Estimating Bias in semi-supervised learning via Nested EM on Gaussian Mixture Models

## Usage 

The `MATLAB` directory contains the MATLAB code used in the orignal paper. 
`main.m` contains an optimization example for a dataset in `dataset.mat` for one initialization.

To run with your own data, specify data matrices `unlabeled`, `labeled_pos`, `labeled_neg` 
with samples as rows and featues as columns. 
Specify the number of components with `num_componets` and an optional progress bar.   
Run optimization with 
```
[alpha, negative_params, positive_params, w_labeled] = ...
    PNU_nested_em(unlabeled, labeled_pos, labeled_neg, num_components, ...
    progress_bar)
```

## Reference

If you use this code for research, please cite our accompaying paper:

```
@inproceedings{depaoliskaluza2022bias,
  title={An Approach to Identifying and Quantifying Bias in Biomedical Data},
  author={De Paolis Kaluza, M. Clara and Jain, Shantanu and Radivojac, Predrag},
  booktitle={Pacific Symposium on Biocomputing 2023: Kohala Coast, Hawaii, USA, 3--7 January 2023},
  pages={311--322},
  year={2022},
}
```

## License

This project is licensed under the MIT license found in the `LICENSE` file in this repository.
