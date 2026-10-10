# PlatesOCR

Automatic reading of Polish license plates with OCR. A convolutional-recurrent neural network (CRNN) trained with the CTC loss reads the text of a plate crop.

The whole pipeline, from downloading the dataset to reading plates, is in one notebook: [PlatesOCR.ipynb](PlatesOCR.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RBorczuch/PlatesOCR/blob/main/PlatesOCR.ipynb)

## Contents

- [PlatesOCR](#platesocr)
  - [Contents](#contents)
  - [Quick start](#quick-start)
  - [Dataset](#dataset)
  - [Pipeline overview](#pipeline-overview)
  - [1. Plate extraction](#1-plate-extraction)
  - [2. Model](#2-model)
    - [Architecture](#architecture)
    - [Why CTC](#why-ctc)
    - [Training model and prediction model](#training-model-and-prediction-model)
  - [3. Training](#3-training)
  - [4. Inference](#4-inference)
  - [Files and folders](#files-and-folders)
  - [Limitations](#limitations)
  - [Previous version](#previous-version)

## Quick start

**Google Colab.** Open the notebook with the badge above, choose a GPU runtime and run all cells. TensorFlow is already installed there.

**Locally.**

```bash
pip install -r requirements.txt
jupyter notebook PlatesOCR.ipynb
```

Two switches in the first code cell choose what runs:


| Switch           | `True`                                                     | `False`                             |
| ---------------- | ---------------------------------------------------------- | ----------------------------------- |
| `RUN_EXTRACTION` | download the dataset (about 600 MB) and cut out the plates | use the existing`plates_images/`    |
| `RUN_TRAINING`   | train a new model, save it to`final_model.h5`              | load the model from`final_model.h5` |

## Dataset

[Poland vehicle license plate dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset) on Kaggle, collected in Katowice by the University of Economics in Katowice (last updated January 2026).


|             |                                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| Photos      | 290, up to 4640 x 2088 px                                                                                          |
| Annotations | `annotations.xml` in CVAT 1.1 format: one box per plate (26 of them rotated) with the plate number as an attribute |
| Plates      | 290 annotations, 267 different plate numbers, mostly 7 or 8 characters                                             |
| Conditions  | sun, clouds, winter, dirty and snowy plates, glare, different angles and distances                                 |
| Download    | public, no Kaggle account needed                                                                                   |

## Pipeline overview

```mermaid
flowchart TD
    K[("Kaggle dataset<br/>290 photos + annotations.xml")]
    EX["1. Plate extraction<br/>rotated box → 4 corners → perspective warp"]
    P[("plates_images/<br/>SK404XK.jpg, STA0461C.jpg, ...")]
    S["Split by plate number<br/>80% training / 20% validation"]
    TR["3. Training<br/>CRNN + CTC loss, augmentation"]
    M[("final_model.h5")]
    EV["Evaluation<br/>plate accuracy, character error rate"]
    INF["4. Inference<br/>read_plates()"]

    K --> EX --> P --> S --> TR --> M
    M --> EV
    M --> INF
    NEW["New plate crops"] --> INF --> TXT["Plate number, e.g. SK404XK"]
```

The model reads plates that are already cut out and straightened. In the dataset the plates are located by the annotations; for new photos a plate detector is needed in front of the model.

## 1. Plate extraction

Each annotation in `annotations.xml` looks like this:

```xml
<image id="3" name="101.jpg" width="2284" height="1522">
  <box label="plate" xtl="1268.64" ytl="679.43" xbr="1631.55" ybr="790.41" rotation="3.70">
    <attribute name="plate number">STA0461C</attribute>
  </box>
</image>
```

Every box is turned into a flat `400 x 100` image named after the plate number:

```mermaid
flowchart LR
    A["CVAT box<br/>xtl, ytl, xbr, ybr"] --> B["Centre and half sizes"]
    B --> C["Add margin<br/>6% of plate height<br/>on every side"]
    C --> D{"rotation<br/>attribute?"}
    D -->|yes| E["Rotate corners clockwise<br/>around the centre"]
    D -->|no| F["Axis-aligned corners"]
    E --> G["4 corners<br/>TL, TR, BR, BL"]
    F --> G
    G --> H["getPerspectiveTransform<br/>+ warpPerspective"]
    H --> I["400 x 100 crop"]
    I --> J["plates_images/STA0461C.jpg"]
```

- The margin makes sure that characters touching the edge of a tight box are not cut off.
- CVAT stores rotation in degrees, clockwise around the box centre. Applying it before the warp straightens tilted plates.
- Spaces in plate numbers are removed (`SB 262FF` → `SB262FF`). The same plate in a second photo is saved as `SB262FF__2.jpg`.

There is no colour filtering or thresholding. An earlier version of this project masked out everything but dark pixels. On this dataset that erased characters on faded and dirty plates, so the network now gets plain grayscale images and learns to handle glare, dirt and shadows itself.

## 2. Model

### Architecture

The model is a CRNN: convolutions turn the image into a sequence of feature vectors (one per column of the plate), and recurrent layers read that sequence from left to right and from right to left.

```mermaid
flowchart TD
    I["Input image<br/>200 x 50 x 1<br/>(transposed: width first)"]
    C1["Conv2D 32, 3x3, ReLU<br/>200 x 50 x 32"]
    P1["MaxPooling 2x2<br/>100 x 25 x 32"]
    C2["Conv2D 64, 3x3, ReLU<br/>100 x 25 x 64"]
    P2["MaxPooling 2x2<br/>50 x 12 x 64"]
    R["Reshape<br/>50 time steps x 768 features"]
    D1["Dense 64, ReLU + Dropout 0.2<br/>50 x 64"]
    L1["Bidirectional LSTM 128<br/>50 x 256"]
    L2["Bidirectional LSTM 64<br/>50 x 128"]
    D2["Dense 37, softmax<br/>50 x 37"]
    O["Probability of each character<br/>+ blank at each of 50 steps"]

    I --> C1 --> P1 --> C2 --> P2 --> R --> D1 --> L1 --> L2 --> D2 --> O
```

The 37 output classes are the 35 characters in `CHARACTERS` (digits and letters except `Q`, which does not occur in the dataset), the `[UNK]` token of `StringLookup` and the CTC blank. The image is transposed before it enters the network, so after the convolutions each of the 50 time steps corresponds to a 4-pixel-wide vertical strip of the plate.

### Why CTC

A plate has 7 or 8 characters, but the network produces 50 predictions, and the annotations do not say where each character is. Connectionist Temporal Classification (CTC) solves this: the loss sums the probability of every alignment of the 50 steps that reduces to the correct text, so the network learns from the plate number alone.

Reading a plate (greedy decoding) reverses that reduction:

```mermaid
flowchart LR
    A["Softmax output<br/>50 steps x 37 classes"] --> B["Most likely class<br/>at every step"]
    B --> C["Merge repeated classes"] --> D["Remove blanks"] --> E["Map indices<br/>to characters"]
```

Example with `-` as the blank:

```
step output:   - - S S - K K - 4 - 0 0 - 4 - - X K K - -
merge repeats: - S - K - 4 - 0 - 4 - X K -
remove blanks: S K 4 0 4 X K   →   "SK404XK"
```

Doubled characters on a plate (such as `11` in `SCZ26114`) survive because the network outputs a blank between them.

### Training model and prediction model

```mermaid
flowchart LR
    subgraph training["Training model (ocr_model_v1)"]
        IMG["image"] --> NET["CNN + BiLSTM<br/>... dense2"]
        LBL["label<br/>(padded with 0)"] --> CTC["CTCLayer<br/>adds CTC loss"]
        NET --> CTC
    end
    subgraph prediction["Prediction model"]
        IMG2["image"] --> NET2["same layers<br/>(shared weights)"] --> OUT["dense2 output"] --> DEC["greedy CTC decoding"] --> T["text"]
    end
    NET -. shared weights .- NET2
```

- Labels in a batch have different lengths, so they are padded with 0 (the `[UNK]` index, which never appears in a real label). `CTCLayer` counts the non-zero entries to get each label's real length.
- `CTCLayer` computes the loss with `tf.nn.ctc_loss` and adds it with `add_loss`, so the model is compiled without a separate loss function. `tf.nn.ctc_loss` and `tf.nn.ctc_greedy_decoder` work with both Keras 2 (TensorFlow 2.15) and Keras 3 (TensorFlow 2.16 and later).
- The prediction model reuses the layers from the input to `dense2`, so it changes together with the training model and with loaded weights.

## 3. Training

```mermaid
flowchart TD
    A["plates_images/*.jpg"] --> B["Label from file name<br/>skip labels with unknown characters"]
    B --> C["Split by plate number (seed 42)<br/>80% / 20% of plate numbers"]
    C --> D["tf.data: read JPEG → grayscale → [0,1]<br/>→ resize 200x50 → transpose<br/>label → indices"]
    D --> AUG["Training set only: random shift,<br/>brightness, contrast, noise"]
    AUG --> PB["padded_batch(16)"]
    D --> PB
    PB --> E["model.fit, Adam, up to 300 epochs"]
    E --> F{"val_loss improved?"}
    F -->|yes| G["Save best_model.weights.h5"] --> E
    F -->|"no for 30 epochs"| H["Early stop,<br/>restore best weights"]
    E -->|"300 epochs done"| H
    H --> I["Save final_model.h5"] --> J["Validation: example predictions,<br/>plate accuracy, CER"]
```

- **Split by plate number.** Some cars appear in more than one photo. All crops of one plate number go to the same subset, so the validation set only contains plates the model has never seen. A random split by image would leak them and make the results look better than they are.
- **Augmentation.** About 230 training images are few for a neural network, which then memorizes the training set. Each epoch, every training image is shifted by a few pixels and gets random brightness, contrast and noise, so the network never sees exactly the same image twice.
- **Metrics.** *Plate accuracy* counts a plate as correct only when every character matches. *Character error rate (CER)* is the number of character edits (Levenshtein distance) between predictions and true plate numbers, divided by the number of true characters. It shows how close the wrong readings are.

With `RUN_TRAINING = False`, `final_model.h5` is loaded with `load_weights` into the same architecture. Loading only the weights avoids rebuilding the custom `CTCLayer` from the file, which is not portable between Keras 2 and Keras 3. If the file was trained with a different `CHARACTERS` list, the notebook stops with an explanation.

## 4. Inference

`read_plates(images)` takes straightened BGR plate crops, such as those produced by `warp_plate()`:

```mermaid
sequenceDiagram
    participant U as Caller
    participant G as cv2.cvtColor
    participant T as to_model_input
    participant M as prediction_model
    participant D as decode_predictions
    U->>G: BGR crops
    G->>T: grayscale crops
    T->>M: batch of 200 x 50 x 1 tensors
    M->>D: 50 x 37 probabilities per plate
    D->>U: ["SK404XK", ...]
```

## Files and folders


| Path                    | Contents                                                                          |
| ----------------------- | --------------------------------------------------------------------------------- |
| `PlatesOCR.ipynb`       | the whole pipeline                                                                |
| `requirements.txt`      | Python dependencies for running locally                                           |
| `dataset/`              | unpacked Kaggle dataset:`annotations.xml` and `photos/` (created by the notebook) |
| `plates_images/`        | straightened plate crops, file name = plate number (created by the notebook)      |
| `best_model.weights.h5` | best weights saved during training                                                |
| `final_model.h5`        | trained model; the committed one is outdated, see[Quick start](#quick-start)      |

## Limitations

- **Small dataset.** 290 photos, 267 different plates. Expect mistakes on unusual plates (custom plates, old black plates, temporary plates). More data helps most. Other Polish datasets, such as Roboflow Universe collections, can be added if they are converted to the `plates_images/` naming scheme.
- **One-row plates only.** Two-row plates (motorcycles, some trailers) do not fit the left-to-right reading of the model.
- **Fixed character set.** A trained model only fits the `CHARACTERS` list it was trained with.

## Previous version

The first version of this project was trained on different data: a license plate dataset from Roboflow Universe, exported in COCO format with a segmentation polygon around every plate. The plates were cut out of the photos, labelled automatically with EasyOCR and corrected by hand, which gave a training set of 250 plates. 

The Roboflow export link used by the notebooks has since expired (the download returns HTTP 403), so the dataset could no longer be downloaded. In this update it was replaced with the [Poland vehicle license plate dataset](#dataset) from Kaggle. Because that dataset already contains the plate numbers, the EasyOCR labelling and manual review steps were removed as well. The original three notebooks (`license_plates_extraction.ipynb`, `PlatesOCR_learning.ipynb`, `PlatesOCR_model_test.ipynb`) are still available in the git history.
