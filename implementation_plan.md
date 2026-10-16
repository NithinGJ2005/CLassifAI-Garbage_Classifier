# Implementation Plan: ClassifAI Model Retraining and Application Integration

## 1. Project Objective

Improve the existing ClassifAI Garbage Classification System by creating a reproducible, high-quality model training workflow and integrating the resulting model into the current Streamlit application.

**Important:** Model training will be performed manually by me in Google Colab. Do not attempt to train the model during implementation. Your responsibility is to prepare the code, Colab notebook, evaluation pipeline, model integration, and documentation.

The primary objective is to improve classification accuracy, per-class precision, recall, macro-F1 score, and generalization to unseen waste images.

## 2. Strictly Preserve the Existing Approach

Do not replace the existing model family or classification approach.

Retain the following technologies:

- Python
- TensorFlow/Keras
- MobileNetV2 with ImageNet-pretrained weights
- Transfer learning followed by optional fine-tuning
- Six-class image classification
- Streamlit for the frontend

Preserve the existing MobileNetV2 architecture and classifier-head design, including Global Average Pooling, a 256-unit Dense layer, Dropout of 0.5, and a six-unit softmax output, after verifying the existing model implementation.

Do not introduce YOLO, Vision Transformers, EfficientNet, a different model family, or a different prediction architecture.

## 3. Dataset

Use this as the primary dataset:

**Dataset:** Waste Classification Harmonized Dataset

**Link:** https://huggingface.co/datasets/TrashBusters/combined

Dataset details:

- Total images: 29,849
- Approximate dataset size: 2.07 GB
- Training split: 20,894 images
- Validation split: 4,477 images
- Test split: 4,478 images
- Number of classes: 6
- License listed on the dataset card: MIT

The dataset combines multiple waste-image sources and includes source metadata that can help evaluate generalization.

The six classes must remain in this exact order throughout training and inference:

1. cardboard
2. glass
3. metal
4. paper
5. plastic
6. trash

Load the dataset directly from Hugging Face inside Google Colab:

```python
!pip install -q datasets scikit-learn matplotlib pandas pillow

from datasets import load_dataset

dataset = load_dataset("TrashBusters/combined")

print(dataset)
print(dataset["train"][0])
```

Inspect the actual dataset metadata, split names, image column, and label mappings before training. Do not assume that the hosted dataset will remain unchanged.

Preserve the supplied training, validation, and test splits. Do not combine them and create a new random split without a compelling, documented reason.

## 4. Google Colab Training Notebook

Create the following notebook in the repository:

`notebooks/ClassifAI_MobileNetV2_Colab.ipynb`

This notebook must be designed to run in Google Colab and execute from beginning to end with minimal manual changes.

### Step 1: Environment setup

- Check the TensorFlow version and available GPU.
- Provide a clear message if the notebook is running without a GPU.
- Set reproducible random seeds.
- Install only the necessary dependencies.
- Avoid version changes that unnecessarily break the standard Colab environment.

I will select a GPU runtime manually in Google Colab.

### Step 2: Dataset inspection

The notebook should:

- Load the dataset and verify all three splits.
- Display the number of images in each split.
- Calculate class distributions.
- Display example images from all six classes.
- Check image dimensions, colour channels, invalid images, and missing labels.
- Verify that the class IDs map correctly to the application’s class names.
- Check for exact duplicate images across splits and report possible leakage.
- Use the source metadata to support additional evaluation by source where practical.

If duplicates across splits are discovered, handle them without moving test examples into the training set. Record any exclusions.

### Step 3: Image preprocessing

Use an input size of 224 × 224 pixels and three RGB channels.

Maintain preprocessing compatible with:

`tensorflow.keras.applications.mobilenet_v2.preprocess_input`

Ensure that pixel normalization is applied exactly once.

Do not apply random augmentation to validation or test images.

Avoid loading the entire dataset as large NumPy arrays into RAM. Implement an efficient batch-based data pipeline with prefetching and appropriate memory management for Google Colab.

### Step 4: Training-time data augmentation

Apply moderate augmentation to the training set only.

Recommended transformations:

- Random horizontal flipping
- Small rotations
- Moderate zoom
- Small translations
- Mild contrast variation

Avoid aggressive transformations that distort the material or make the waste category ambiguous.

Inspect some augmented samples before starting full training.

The dataset is approximately balanced, so do not automatically introduce class weights or oversampling. First inspect the actual class distribution and use these techniques only if justified.

### Step 5: Phase One — Train the classifier head

Initialize MobileNetV2 with ImageNet-pretrained weights and freeze its backbone.

Train the existing classifier head using:

- Dense layer: 256 units with ReLU
- Dropout: 0.5
- Output: six classes with softmax
- Optimizer: Adam
- Initial learning rate: 0.001
- Maximum epochs: 20
- Batch size: 32, with the ability to reduce it to 16 if GPU memory is insufficient

Use categorical or sparse categorical cross-entropy consistently with the label format.

Add the following callbacks:

- EarlyStopping with restored best weights
- ReduceLROnPlateau
- ModelCheckpoint

Monitor validation loss for early stopping and checkpointing. Save the best model checkpoint from this phase.

### Step 6: Phase Two — Fine-tune MobileNetV2

Start from the best checkpoint obtained in Phase One.

Gradually unfreeze the final 30 layers of MobileNetV2, while keeping Batch Normalization layers frozen. Use a lower learning rate, initially around 0.00001.

Recommended configuration:

- Maximum epochs: 15
- Learning rate: 0.00001
- Batch size: 16 or 32 depending on GPU memory
- Early stopping and model checkpointing enabled

Keep the classifier-head architecture unchanged.

Compare the fine-tuned model with the frozen-backbone checkpoint using validation metrics. Select the candidate using validation performance, giving priority to macro-F1 alongside validation loss and per-class recall.

Fine-tuning should be retained only if it gives a meaningful improvement without introducing instability or overfitting.

These are starting hyperparameters, not guaranteed optimal values.

## 5. Evaluation and Performance Reporting

Create a complete evaluation pipeline.

Calculate the following metrics:

- Overall accuracy
- Precision for each class
- Recall for each class
- F1-score for each class
- Macro precision, macro recall, and macro-F1
- Weighted-F1 score
- Balanced accuracy
- Test loss
- Confusion matrix
- Average inference time per image

Also generate:

- Training and validation loss curves
- Training and validation accuracy curves
- A per-class classification report
- Samples of misclassified images with their true labels and predictions
- Performance breakdowns by source dataset or image context when practical

Use the validation set for model selection and tuning.

Use the test set for final evaluation only after the training configuration and candidate model have been selected. Do not repeatedly tune the model against test results.

If the existing model is available, support an optional fair comparison between the old and new models on the same held-out test images and preprocessing pipeline.

Do not hardcode, fabricate, or promise an accuracy percentage. Report the actual evaluation results generated by Colab.

## 6. Model Export and Deliverables

At the end of training, export the selected model in Keras format:

`garbage_classifier.keras`

Also generate:

- `class_names.json`
- `test_metrics.json`
- `classification_report.csv`
- `confusion_matrix.png`
- `training_curves.png`
- `training_history.csv`
- A short evaluation summary describing the final results

Ensure that the class-name order in `class_names.json` matches the six-class output layer exactly.

Add a final notebook cell that saves the artifacts to a designated Google Drive folder, after I explicitly mount Google Drive, or packages them for download.

Do not commit a large trained model or overwrite the existing production model automatically.

## 7. Streamlit Application Integration

Keep the existing ClassifAI interface and features.

The existing application must continue to provide:

- Image upload
- Waste classification
- Prediction confidence display
- Carbon-footprint awareness information
- SDG-linked information
- Six-class output

Update the model-loading logic to prefer the newly trained model at:

`artifacts/models/garbage_classifier.keras`

If the new file is not present, optionally fall back to the existing `garbage_classifier.h5` model so the application remains usable during migration.

Preserve the existing RGB conversion, 224 × 224 resize, and MobileNetV2 preprocessing. Do not preprocess the image twice.

Verify that model predictions are mapped correctly to the six class names.

Add a configurable low-confidence warning so uncertain predictions can be reviewed. Choose or validate the warning threshold using validation data where possible. Clearly label the result as model confidence, not guaranteed prediction correctness.

Keep the existing interface design unless a minimal change is needed for compatibility or to display useful evaluation information.

Do not add unsupported claims that the system has measurably improved recycling outcomes or reduced carbon emissions.

## 8. Testing and Acceptance Criteria

Before considering the integration complete, verify that:

1. The application loads the new model successfully.
2. All six classes produce correctly mapped outputs.
3. The existing preprocessing matches the training pipeline.
4. The model can classify sample images without shape or loading errors.
5. The confidence score is displayed correctly.
6. Missing model files produce a useful error message or fall back to the old model.
7. Existing Streamlit features continue to work.
8. Training artifacts and evaluation reports are generated correctly.
9. No test data has leaked into training.
10. Results are reproducible as far as practical, given the Colab environment and GPU nondeterminism.

Compare the new model with the original baseline. The desired outcome is an evidence-based improvement in validation macro-F1 and class-specific performance, followed by transparent final test results. If the new model does not improve, document the findings rather than claiming success.

## 9. Required Repository Changes

Implement and document these deliverables:

- `notebooks/ClassifAI_MobileNetV2_Colab.ipynb` — complete Colab training notebook
- `TRAINING_IN_COLAB.md` — detailed setup, training, evaluation, export, and download instructions
- Updated Streamlit model-loading logic, while retaining backward compatibility
- Model artifact directory instructions
- Tests for class mapping, image preprocessing, model input/output shapes, and model loading
- Evaluation report generation

If the repository already contains an equivalent file or test, improve it rather than creating unnecessary duplicates.

Do not train the model yourself. Do not replace the deployed model before I manually train, evaluate, download, and approve the new model in Google Colab.

## 10. Expected Workflow

The completed workflow should be:

1. Integravity prepares the notebook, code, application integration, and documentation.
2. I open the notebook in Google Colab and select a GPU runtime.
3. I run the notebook to train and evaluate MobileNetV2.
4. I review the metrics and compare the candidate with the existing model.
5. I download the selected model and its class mapping.
6. I place the approved model in `artifacts/models/`.
7. I test the Streamlit application locally and then deploy it.
8. I retain a backup of the previous model for rollback.

The final implementation must make training repeatable, evaluation trustworthy, and model replacement safe without changing the existing model family or the purpose of ClassifAI.