## Project: Human Image Segmentation using Pytorch and U-Net Architecture
This repository focuses on human image segmentation using PyTorch and the U-Net architecture, which is a popular choice for such tasks. The main objective is to accurately identify areas where humans are present and generate corresponding segmentation masks, marking these areas as white while the rest of the image remains black.

### Project Structure
The project follows a structured approach, outlined as follows:

1. **Configuration Setup**: Define configurations essential for the project and visualize sample images to understand the dataset.
  
2. **Dataset Splitting**: Split the dataset into training and validation sets to facilitate model training and evaluation.
  
3. **Data Augmentation**: Implement data augmentation techniques tailored for both the training and validation sets. In segmentation tasks, augmentation applies to both images and their corresponding masks.
  
4. **Custom Dataset Creation**: Create a custom dataset to handle image-mask pairs, ensuring proper alignment between images and their masks.
  
5. **Data Loading**: Utilize DataLoader to efficiently load the dataset into batches for training.

6. **Model Building**: Build the segmentation model using the U-Net architecture. We'll leverage the Segmentation Models library in PyTorch, exploring various loss functions like Dice Loss and BCE with Logits Loss.

7. **Training and Validation**: Define training and validation functions, and implement the training loop to train the model on the dataset.

8. **Model Deployment**: Deploy the best-performing model to create an interface for segmentation tasks, enabling easy use in real-world scenarios.

### Requirements
- Python 3.x
- PyTorch
- Segmentation Models Library
- NumPy
- Pandas


### Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

### Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Segmentation Models Library](https://github.com/qubvel/segmentation_models.pytorch)



### Contact
For any inquiries or support, please contact [za.shahlaie@gmail.com](mailto:za.shahlaie@gmail.com).

### References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Binary Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
- [Dice Loss](https://github.com/qubvel/segmentation_models.pytorch#losses)
- [BCE with Logits Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
