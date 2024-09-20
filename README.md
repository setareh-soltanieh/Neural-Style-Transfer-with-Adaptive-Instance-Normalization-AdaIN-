# Neural Style Transfer with Adaptive Instance Normalization (AdaIN)

This project implements neural style transfer using an Adaptive Instance Normalization (AdaIN) network. The model uses a pre-trained VGG network as the encoder and trains the decoder by optimizing style and content loss. The network is trained on the COCO1K and COCO10K datasets, and different style images are applied to content images to evaluate the results.

## Key Features:
- **Training**: The network is trained on two datasets (COCO1K and COCO10K) for up to 1000 epochs, with model weights saved at regular intervals.
- **Hyperparameters**: Learning rate is dynamically adjusted across epochs, and experiments are run with varying batch sizes and alpha values to observe the impact on style transfer.
- **Style Transfer Results**: Tested on various well-known style images like Andy Warhol's works and Chagall’s paintings.
- **Implementation**: Python scripts (`my_train_1k.py` and `my_train_10k.py`) are provided for training on 1K and 10K datasets. Pre-trained model weights for both datasets are included.

This project demonstrates how adjusting parameters like alpha and gamma influences the balance between style and content in the output images.

---

## How to Run:
1. Clone the repository.
2. Install the required dependencies from the `requirements.txt` file.
3. Run `my_train_1k.py` or `my_train_10k.py` depending on the dataset you wish to train on.

## Results:
The results showcase how the content images adopt the styles of the respective style images. The model's performance improves when trained on larger datasets like COCO10K, with more accurate color representation and fewer artifacts.

Pre-trained weights for both the 1K and 10K datasets are provided in the respective `model_1k` and `model_10k` folders.

---

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
