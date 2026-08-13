# Machine Learning for High-Performance Optical Sorting

Comparing a rule-based baseline, two linear models, a Random Forest and a Multi-Layer Perceptron for automated waste sorting on the TrashNet dataset, scored on what a sorting plant actually buys: **material purity, yield, recovery, robustness to sensor degradation, and inference latency against a 50 ms budget.** A fine-tuned CNN is included as a reference point on all three axes.

**Author:** Mohamed Tawfeek · **Supervisor:** Dr. Nick Hay · University of Sussex BSc final-year project.

The classical pipeline is interpretable, since feature importances say which part of the representation does the work; it trains in seconds on a CPU with no accelerator anywhere in the loop; and it is small enough to reason about end to end. What it is not is the faster or the more robust option, and the sections below are as much about where it fails as where it holds. The value here is in the relative comparisons and the failure analysis rather than any single accuracy figure.

## What the project establishes

**Industrial metrics.** Every model is scored on material purity, yield and recovery, the precision and recall pair the sensor-based sorting literature tunes and trades off against each other, and against a 50 ms latency budget derived from a real conveyor throughput target, not on accuracy alone.

**Robustness.** Clean accuracy does not settle the deployment question. Under simulated sensor degradation the ordering reverses: the Random Forest beats the more accurate MLP under motion blur and low light, so the right model is conditional on the imaging environment.

**Latency.** The Random Forest as trained misses the 50 ms budget the same weights meet single-threaded, because parallelising 200 trees over one 122-feature prediction costs more in scheduling overhead than it saves.

**Self-audit.** The CNN's headline 95.00% is the best of its five seeds; across all five it is 92.16% ± 1.80, and the conclusions that number was used to support are revised to the five-seed reading rather than left on the flattering one.

**Demo.** The demo runs on a clean clone with no dataset download and no training pass, and puts a glass bottle and a tin can side by side that both classifiers get exactly backwards.

## Results

Test set, 380 images. The last row is a reference ceiling, not a sixth candidate.

| Model | Accuracy | Macro F1 | Purity | Yield | Recovery |
|---|---|---|---|---|---|
| Rule-based HSV baseline | 24.50% | 0.1749 | 0.2465 | 0.2152 | 0.1011 |
| Linear SVC | 67.11% | 0.6620 | 0.6742 | 0.6540 | 0.5082 |
| Logistic Regression | 68.95% | 0.6687 | 0.6841 | 0.6591 | 0.5141 |
| Random Forest | 78.95% | 0.7747 | 0.7993 | 0.7612 | 0.6381 |
| Multi-Layer Perceptron | 80.53% | 0.7914 | 0.7965 | 0.7876 | 0.6605 |
| *ResNet18 (reference)* | *95.00%* | *0.9408* | *0.9517* | *0.9324* | *0.8899* |

Purity is precision, Yield is recall, Recovery is the composite TP/(TP+FP+FN) that penalises output contamination and lost material in one number. The CNN row is the most favourable of five seeds and should be read with the spread notebook 17 measures: 92.16% ± 1.80 accuracy, still twelve points clear of the MLP's 80.05% ± 1.88. The accuracy ordering is unaffected; individual error counts drawn from that row are, and finding 1 says where.

![Accuracy, F1 and the three industrial metrics across all models](results/figures/master_comparison_all.png)

The two linear rows are not deployment candidates. They exist to split the 54-point jump from baseline to Random Forest into its causes: how much belongs to the descriptor, how much to learning the boundary at all, and how much to non-linearity. The descriptor is the largest single step, worth about half the total.

---

## Three findings worth reading the code for

### 1. The feature space, not the classifier, sets the accuracy ceiling

The dominant error in every classical model is a near-symmetric glass to metal confusion. Four model families with very different inductive biases fail at almost the same rate on the same pair, which points at the shared representation rather than at any one model. Transparent glass lets the grey background dominate its histogram, and metal is specular grey, so both land in the same low-saturation region of the feature space. A fine-tuned CNN roughly halves the absolute count on the pair, but the audit shows it does not reduce glass and metal's *share* of the errors that remain.

<details>
<summary><b>The full analysis, including the self-audit</b></summary>

The glass and metal off-diagonal is the largest error for all four classical models: 16 images for the RF, 17 for the MLP, 21 for logistic regression, 25 for the linear SVC. The weaker the model, the worse it breaks, and the linear models turn glass into an attractor rather than trading images evenly. Glass and metal are also the two weakest classes by F1 for every model, while cardboard and paper, which colour alone might be expected to confuse, sit at 0.86 and above.

![Confusion matrices for the linear baselines](results/figures/linear_baseline_confusion_matrices.png)

The CNN moves the pair, and notebook 17 audits how far. A fine-tuned ResNet18 puts only 2 images on that off-diagonal at seed 42. Reported on its own that reads as the descriptor being the whole story, but 2 is the best of five seeds; across seeds 42, 1, 7, 13 and 99 the count is 7.6 ± 4.4, with a worst seed inside the range the classical models occupy. The number that carries the point is the pair as a share of total errors:

| Model | pair total | total errors | pair as share of errors |
|---|---|---|---|
| Linear SVC | 25 | 125 | 20.0% |
| Logistic Regression | 21 | 118 | 17.8% |
| Random Forest | 16 | 80 | 20.0% |
| MLP | 17 | 74 | 23.0% |
| *ResNet18 fine-tuned, seed 42* | *2* | *19* | *10.5%* |
| *ResNet18 fine-tuned, five seeds* | *7.6 ± 4.4* | *29.8 ± 6.8* | *24.2% ± 11.8* |

The CNN's absolute count falls because its total errors fall, from 80 and 74 down to about 30, not because glass and metal became relatively easier. Across seven representations, from three hand-set thresholds to a fine-tuned convolutional stack, this one pair accounts for between about a fifth and a quarter of the errors every time. So the section has two claims and the audit separates them. The first, that the representation and not the classifier sets the ceiling, survives: four classifiers on the same 122 dimensions land between 67% and 81%, and changing the representation is worth twelve points where changing the classifier family was worth ten at most. The second, that glass and metal is an artifact of the HSV and LBP descriptors specifically, does not survive the share column, since nothing tried here made the pair a smaller fraction of the problem.

I also checked whether the CNN's edge is memorised near-duplicates. TrashNet photographs the same object from several angles, so an image-level split leaks. Measured on frozen ImageNet features, which were never fitted to these six classes and are independent of everything being compared, the CNN's margin over the MLP holds across five similarity cutoffs, and both of its residual glass and metal errors sit on the least similar test images, which is the opposite of what memorisation would look like.

![ResNet18 confusion matrix, glass and metal reduced to one image each way](results/figures/cnn_confusion_matrix.png)

</details>

### 2. Clean accuracy does not decide the deployment question

Four failure modes a conveyor sensor plausibly produces, motion blur, Gaussian noise, brightness reduction and JPEG compression, applied to the raw image before feature extraction, at five severities each. The ranking from clean data does not survive them. Under motion blur the Random Forest is 14.7 points ahead of the MLP, a model it cannot be separated from on clean data, with brightness reduction showing the same reversal from the mildest severity onward. So model choice within the classical pipeline is conditional: the MLP in a controlled enclosure for its latency headroom, the RF where blur or illumination cannot be controlled.

<details>
<summary><b>The full analysis</b></summary>

Accuracy at worst-case severity, with chance on six classes at 0.167:

| Perturbation | Random Forest | MLP | *ResNet18 (ref.)* |
|---|---|---|---|
| Motion blur | 0.658 | 0.511 | *0.813* |
| Gaussian noise | 0.234 | 0.168 | *0.368* |
| Brightness reduction | 0.418 | 0.358 | *0.492* |
| JPEG compression | 0.253 | 0.300 | *0.618* |

Two of these rows are noise. At worst-case Gaussian noise both classical models have already failed and the RF's lead is a lead inside a region where nothing works; the JPEG margin between them is small enough to ignore. The row that carries real information is motion blur.

![Perturbation examples](results/figures/perturbation_examples_improved.png)

Noise is a cliff rather than a slope, and the descriptor is why. The colour histograms stop being class-specific as pixels scatter across bins, each channel in its own way, while the LBP histogram fails in the opposite direction, collapsing into its non-uniform catch-all bin because noise breaks the smooth pixel transitions that produce uniform codes. Notebook 16 turns that into a testable prediction: a model that does not use the descriptor should not fall off the cliff. At the mildest noise tested the RF drops from 0.789 to 0.342 and the MLP from 0.805 to 0.379, while the ResNet18 goes from 0.950 to 0.926. The classical pipeline loses more than half its accuracy to a perturbation that costs the CNN two points, which makes noise sensitivity a descriptor problem first and a hardware problem second.

![Feature-vector shift under Gaussian noise](results/figures/gaussian_noise_feature_shift.png)
![Accuracy against severity for all four families](results/figures/cnn_robustness_curves.png)

</details>

### 3. Feature extraction, not the classifier, owns the latency budget

Feature extraction costs 18.2 ms and is identical in every pipeline, so 36% of the 50 ms budget is spent before any classifier runs. For the MLP the classifier is a rounding error at 0.10 ms, which means the descriptor is effectively the whole pipeline. The Random Forest exposes a threading trap: as trained it takes 57.8 ms end to end and misses the budget, while the identical weights at `n_jobs=1` meet it at 27.7 ms, because parallelising 200 trees for one prediction costs more in scheduling overhead than it saves.

<details>
<summary><b>The full analysis</b></summary>

Single-image CPU latency, end to end:

| Configuration | End-to-end | Within 50 ms? |
|---|---|---|
| RF, `n_jobs=1` | 27.7 ms | yes |
| RF, `n_jobs=-1` (as trained) | 57.8 ms | no |
| MLP | 18.5 ms | yes |
| *ResNet18 (ref.), 8 threads* | *20.3 ms* | *yes* |

A 2 m/s belt at 10 cm spacing needs 1,200 items/min. The RF at `n_jobs=1` clears it at roughly 2,160/min, the MLP does comfortably at roughly 3,240/min, and the RF as trained does not at roughly 1,040/min.

The CNN's 20.3 ms is real but it is a whole-machine number, and it has to be read as one. `src/features.py` is a single-threaded Python loop, so the 18.2 ms descriptor gets one core while the ResNet18 gets eight. Matched core for core the ordering reverses: the single-thread forward pass is 48.6 ms against the descriptor's 18.2 ms, and the CNN's single-thread end to end of 51.2 ms misses the budget the descriptor pipeline meets. Per core the handcrafted descriptor is roughly two and a half times cheaper, and the CNN only fits the budget by spending cores the classical pipeline never asks for.

The obvious optimisation route is now measured and closed. Batching does nothing for extraction, because `src/features.py` is a per-image loop and costs 18.3 ms per item whether it is handed 1 image or 128. That leaves a compiled implementation or a cheaper descriptor, and since the LBP computes a 24-neighbour comparison at every pixel in Python-level scikit-image code, that is where the cost sits.

![Latency and throughput against batch size](results/figures/cnn_latency_throughput.png)
![RF latency across thread counts](results/figures/thread_count_latency.png)

</details>

---

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook demo/demo_classifier.ipynb
```

The trained models and extracted features are committed, and so are the three images the demo classifies, under `demo/images/`. The demo therefore runs on a clean clone with no dataset download and no training pass. It loads an image, shows its HSV and LBP feature histograms, and runs both classifiers with confidence scores. It also shows the glass bottle and tin can both models get exactly backwards, so finding 1 is visible directly rather than taken on trust.

Every other notebook needs the full dataset. See the collapsibles below.

---

<details>
<summary><b>Method</b></summary>

**Features (122 dimensions).** Images are resized to 224×224 and converted to HSV. Three 32-bin normalised channel histograms give 96 colour dimensions; a uniform Local Binary Pattern descriptor (radius 3, 24 points) gives 26 texture dimensions. All extraction lives in `src/features.py` as the single source of truth, and notebook 02 asserts the module reproduces its step-by-step derivation exactly, so training and evaluation cannot drift apart.

**Split.** 2,527 images, stratified 70/15/15 into 1,769 train, 378 validation and 380 test, `random_state=42` throughout. Hyperparameters were selected by 5-fold stratified cross-validation on the training partition only; the test set was used only after configurations were fixed. Where cross-validation means were within noise of each other I took the configuration with the lowest fold-to-fold standard deviation rather than the highest mean.

**Models.** A hand-tuned HSV threshold classifier as a lower bound; a logistic regression and a linear SVC, both C tuned by the same cross-validation and both wrapped with a StandardScaler refitted inside each fold; a Random Forest of 200 trees at unbounded depth; an MLP with one hidden layer of 200 units, alpha 0.01, scaler fitted on training data only.

**CNN reference (notebook 16).** An ImageNet-pretrained ResNet18 with the 1000-way head replaced by a 6-way one, all layers unfrozen, trained for 20 epochs of AdamW at learning rate 1e-4 with cosine decay and batch size 32, with the epoch chosen on the same validation partition. It sees the same pixels the descriptor does, through the identical resize call, so no part of the difference comes from a different resize path. The only augmentation is a horizontal flip, which cannot manufacture the robustness result since none of the four perturbation families are geometric.

**Auditing the CNN (notebook 17).** A frozen ImageNet probe and a from-scratch ResNet18 separate the contribution of pretraining from that of the TrashNet images, each swept over the same five seeds. A near-duplicate audit ranks test images by similarity to their nearest training image, sweeps five cutoffs rather than picking one, and re-scores on each cleaned subset without retraining, run once in the fine-tuned space and once in the independent frozen one. A five-seed sweep matching notebook 15 measures how much of the seed-42 result is the seed. Throughout, the glass and metal pair is reported as a share of each model's errors alongside the absolute count.

**Industrial metrics.** Purity is precision, Yield is recall, both standard in the sensor-based sorting literature (Küppers et al. 2021; Maier et al. 2024). Recovery is the composite TP/(TP+FP+FN).

**Feature importance.** Block permutation importance on the test partition puts texture as the largest single group at 32.6% for the RF, revising the near-even split that mean impurity decrease reports, since MDI is measured on data the trees have already fitted and favours features offering more split points. That makes the case for fusing colour with texture an empirical result on this dataset rather than an argument from prior work.

</details>

<details>
<summary><b>Reproducing from scratch</b></summary>

Download TrashNet from https://github.com/garythung/trashnet and extract so that:

```
datasets/dataset-resized/{glass,paper,cardboard,plastic,metal,trash}/
```

Then run the notebooks in numerical order. 01 and 02 build the feature matrix, 03 to 05 train and save the models, and 06 onward depend on those artifacts. The robustness sweep in notebook 06 re-extracts features 21 times and takes roughly 20 minutes. The CNN notebooks 16 and 17 need PyTorch, which the rest of the project does not, and take roughly half an hour each on CPU.

`requirements.txt` pins scikit-learn to the minor version the committed pickles were written with, since scikit-learn does not guarantee pickle compatibility across releases. The CNN weights (`cnn_final_model.pt`, 43 MB) are gitignored and regenerated by notebook 16; everything scored from them is committed as CSV, JSON and `.npy`. Run notebook 16 before 17.

**Repository layout:**

```
src/         features.py · metrics.py · perturbations.py · split.py   canonical, imported everywhere
notebooks/   01–17, numbered in dependency order
demo/        demo_classifier.ipynb and the three images it classifies
results/     committed pickles, confusion matrices, metrics and figures
```

</details>

<details>
<summary><b>Technical notes</b></summary>

**Class ordering.** scikit-learn orders classes alphabetically, which is not the order of the `CLASSES` list. Anything labelling a confusion matrix, classification report or per-class table uses `CLASS_ORDER = sorted(CLASSES)` and passes it as `labels=`, since passing an unsorted list silently mislabels every per-class result without raising: the support column still sums and the diagonal still looks like a diagonal. An earlier version of this analysis identified the wrong pair for exactly this reason, and the assert now prevents it.

**Resize filter.** LBP shifts about 1.6% under a change of resize filter alone, roughly three times more than the colour histograms, because it measures local pixel contrast. The pipeline pins Pillow's default bicubic filter and asserts it, after an early version used different filters in different stages.

**CNN reproducibility.** Seed-42 retraining recovers the committed model exactly. Scoring the committed weights is device-sensitive: on CPU, or on CUDA at batch 64, one borderline test image flips and accuracy reads 94.74% rather than 95.00%. Neither number is wrong; one image sits close enough to a decision boundary that the choice of convolution algorithm decides it, which is worth knowing before reading anything into small error counts.

**Transferring the split across representations.** `split_indices` in `src/split.py` returns the same partition as positional indices, so a raw-image model is evaluated on exactly the same test images as a feature-matrix model. Notebook 16 asserts this rather than relying on the reasoning.

</details>

---

## How to read the numbers

TrashNet is single objects on a light background under controlled lighting. Real belt imagery has overlap, occlusion, contamination and motion. Everything above is an upper bound on deployment performance, not an estimate of it, and the worth of the project is in the relative comparisons and the failure analysis rather than the absolute accuracies.

## References

Friedrich et al. (2022), *MethodsX* 9, 101686. Küppers et al. (2021), *Waste Management & Research* 39(1). Maier et al. (2024), *IEEE Access* 12. Mesina et al. (2007), *International Journal of Mineral Processing* 82(4). Smith et al. (2019), *Minerals Engineering* 133. Thung & Yang (2016), TrashNet, CS229 project report, Stanford University.

## Dataset and licence

TrashNet, 2,527 images across six classes, from Thung & Yang (2016), https://github.com/garythung/trashnet. Not redistributed here, apart from the three images under `demo/images/` that keep the demo self-contained; attribution to Thung & Yang (2016) is unchanged.

Code released under the MIT Licence. See [LICENSE](LICENSE).
