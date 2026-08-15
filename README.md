# MNIST Neural Network from Scratch

This project demonstrates how to build and evaluate a fully connected neural network for handwritten-digit classification using PyTorch tensors and manually managed parameters.

The complete workflow is presented in [`nn.ipynb`](nn.ipynb) as a tutorial-style notebook.

## What the Notebook Covers

- Downloads the MNIST dataset with Kaggle Hub.
- Parses the raw IDX image and label files with a custom data loader.
- Explores dataset sizes and image dimensions.
- Converts, flattens, and normalizes the image tensors.
- Initializes a `784 -> 256 -> 128 -> 10` neural network.
- Implements the forward pass, cross-entropy loss, backpropagation, and gradient-descent updates.
- Evaluates predictions with a classification report and confusion matrix.
- Saves and reloads the learned weights.

## Project Files

| File | Description |
| --- | --- |
| `nn.ipynb` | Main tutorial and executable experiment. |
| `app.py` | Local web server and PyTorch inference endpoint. |
| `index.html` | Interactive 28 x 28 digit-drawing interface. |
| `test_app.py` | Inference input and output contract checks. |
| `README.md` | Project overview and setup instructions. |
| `.gitignore` | Excludes local environments, generated models, data, caches, and agent configuration. |

Generated files such as `model_weights.pth` are intentionally excluded from Git.

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install jupyter kagglehub numpy matplotlib torch scikit-learn
```

The dataset is downloaded automatically when the first section of the notebook runs.

## Run the Notebook

```powershell
jupyter notebook nn.ipynb
```

You can also open `nn.ipynb` directly in an editor with Jupyter support, such as Visual Studio Code.

Run cells from top to bottom so that the dataset, tensors, parameters, and predictions are created in dependency order.

## Run the Digit Lab

First run the notebook through the model-saving cell so that `model_weights.pth` exists. Then start the local app:

```powershell
.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), draw a digit, and lift the pointer to see the model's prediction. The visible drawing surface is enlarged for usability, but its internal resolution remains exactly 28 x 28 pixels.

Run the inference checks with:

```powershell
.venv\Scripts\python.exe -m unittest test_app.py
```

## Notes

- Parameters and random preview images are not seeded, so results can vary between runs.
- The live training-loss chart is updated frequently and may make training slower.
- Existing notebook outputs are preserved for reference, including an interrupted training run and its later evaluation results.
