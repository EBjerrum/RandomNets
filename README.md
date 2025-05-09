# RandomNets: A Vectorized Approach to Implicit Ensemble Neural Networks

<div align="center">
<img src="https://raw.githubusercontent.com/EBjerrum/RandomNets/main/resources/RandomNets_HeroImg.jpg" width="500">
</div>
                         
RandomNets is an efficient, vectorized solution for creating and training implicit ensemble feed-forward neural networks, designed to enhance prediction robustness and performance. By introducing input feature masking, RandomNets generates diverse predictions within a single model architecture, making it an attractive alternative to standard feed-forward neural networks. The model is especially suited for molecular property prediction tasks and has demonstrated superior performance across a wide range of bioactivity datasets.

## Key Features

- **Implicit Ensembles**: Achieves ensemble performance with minimal computational overhead by leveraging vectorization.
- **Feature Masking**: Adds diversity to predictions, optimizing performance with deterministic inference.
- **Scalability**: Training time grows sublinearly with ensemble size, enabling efficient scaling to larger ensembles.
- **Open Source**: Provided under an LGPL license for easy adoption and adaptation.

## Installation

To install the RandomNets package, you can use pip with the following command:

```bash
pip install git+https://github.com/EBjerrum/randomnets
```

## Usage

After installation, you can quickly set up and train a RandomNets model for your molecular property prediction tasks using the Pytorch Lightning compatible FpsDataModule and RandomNets model.

I've put together a small tutorial in [this Notebook](https://github.com/EBjerrum/RandomNets/blob/main/notebooks/training.ipynb)

Refer to PyTorch lighning documentation for training and inference.

## Citation

If you use RandomNets in your work, please cite the associated preprint available on ChemRxiv:

Bjerrum, E. J. (2024). RandomNets Improve Neural Network Regression Performance via Implicit Ensembling. ChemRxiv.
[https://chemrxiv.org/engage/chemrxiv/article-details/67656cfa81d2151a02603f48](https://chemrxiv.org/engage/chemrxiv/article-details/67656cfa81d2151a02603f48)

The paper contains details on approach, implementation, testing and results from 133 bioactivity datasets.

## Contributing

Contributions, bug reports, and suggestions are welcome! Please open an issue, or reach out (I'm googlable) or submit a pull request to improve the project.
I recommend reaching out first to avoid duplication of work.

## License

This project is licensed under the LGPL license. See the `LICENSE` file for details.

## Support

For commercial support or consultancy, please contact [Cheminformania Consulting](https://www.cheminformania.com/cheminformania-consulting/)
