
# Face Landmarks Detection Using ResNet18

## Project Overview
This project focuses on detecting facial landmarks using a deep learning approach. The model is trained on the **IBUG 300-W dataset**, which contains images annotated with 68 facial landmarks. We leverage **ResNet18**, a convolutional neural network architecture, and modify it for this task by adjusting the input and output layers to handle grayscale images and produce the 136 facial landmark coordinates (68 points, each with an X and Y coordinate).

## Dataset
The **IBUG 300-W Large Face Landmark Dataset** consists of over 6,666 images of human faces, each annotated with 68 landmark points corresponding to key facial features (eyes, nose, mouth, etc.).

### Downloading the Dataset
The dataset is downloaded and extracted using a script in Python:

```python
!wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
!tar -xvzf 'ibug_300W_large_face_landmark_dataset.tar.gz'
```

### Visualizing Data
Before training, the dataset is visualized to better understand the input images and the labeled landmarks. For example, an image and its corresponding landmark points are plotted using `matplotlib`.

## Data Preprocessing
### Transformations
A `Transforms` class was implemented to apply various augmentations to the images and their landmarks, which helps improve the generalization of the model. The following transformations were applied:
- **Rotation**: Random rotations within a small range.
- **Resize**: Resize all images to 224x224 pixels for compatibility with ResNet18.
- **Color Jitter**: Small perturbations in brightness, contrast, saturation, and hue to make the model robust to lighting variations.
- **Cropping**: Crop images based on bounding boxes provided in the dataset, isolating the face region.

### Dataset Class
A custom dataset class, `FaceLandmarksDataset`, was created to load the images and landmarks, apply the necessary transformations, and return them as tensors for the model. The dataset is split into a training set (90%) and a validation set (10%).

## Model Architecture
For the face landmark detection task, the **ResNet18** model from `torchvision` was modified as follows:
- **Input Layer**: Changed the input layer to accept grayscale images (1 channel) instead of RGB images (3 channels).
- **Output Layer**: The output layer was changed to produce 136 values (68 facial landmarks, each with X and Y coordinates).

```python
self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.model.fc = nn.Linear(self.model.fc.in_features, num_classes=136)
```

## Loss Function and Optimization
The model was trained using **Mean Squared Error (MSE)** loss, which computes the difference between the predicted and actual landmark positions. The **Adam** optimizer was used for updating model weights with a learning rate of 0.0001.

## Training Process
The network was trained for **10 epochs** with a batch size of 64 for the training set and 8 for the validation set. The training loss and validation loss were tracked to monitor performance during training. The model with the lowest validation loss was saved to disk.

```python
torch.save(network.state_dict(), '/content/face_landmarks.pth')
```

### Helper Functions
A helper function was implemented to print the training and validation progress, providing real-time updates on the loss after each batch.

## Testing and Inference
After training, the model was tested on the validation dataset. For each image in the validation set, the predicted and true landmark points were plotted to visually assess the model's performance. The model was able to detect the facial landmarks with reasonable accuracy.

### Example Visualization
The green points represent the true landmarks, and the red points are the predicted landmarks. The visual comparison allows us to gauge the quality of the model's predictions.

## Results
- The model achieved low validation loss, indicating that it can accurately detect facial landmarks on unseen images.
- The predicted landmarks closely follow the true landmarks, as observed in the visualizations.

## Conclusion
This project demonstrates the successful application of ResNet18 for detecting facial landmarks. The dataset was preprocessed with various augmentations to improve model robustness. The trained model generalizes well on unseen images, as seen from the accurate landmark predictions.
