# NYC Airbnb Price Prediction Pipeline

This is my project for the WGU/Udacity MLOps course. The goal was to build a machine learning pipeline that predicts short-term rental prices in New York City, using a Random Forest model.

## Links

- **GitHub Repo:** https://github.com/soniaccodes/Project-Build-an-ML-Pipeline-Starter
- **Weights & Biases Project:** https://wandb.ai/soniaccodes-western-governors-university/nyc_airbnb

## What This Project Does

Instead of just training one model in a notebook, this project builds a **pipeline**. A series of connected steps that can be run again and again, even on new data, without having to redo everything by hand.

The tools I used:
- **MLflow** to run each step and keep everything organized
- **Hydra** to manage settings/config (like hyperparameters) without hardcoding them
- **Weights & Biases (W&B)** to track experiments and store data/model files
- **scikit-learn** to actually train the model (a Random Forest Regressor)

## The Steps

1. **Get the data** download a sample of listings and save it to W&B
2. **Clean the data** remove weird prices, fix date formatting, and filter out listings with locations outside NYC
3. **Test the data** run automated checks to make sure the cleaned data looks right before training on it
4. **Split the data** divide it into training, validation, and test sets
5. **Train the model** train a Random Forest to predict price, and save the trained model
6. **Test the model** check how the model performs on data it's never seen, to make sure it's not overfitting

## How to Run It

Run the whole pipeline:
```bash
mlflow run .
```

Run just one step (ex: data cleaning):
```bash
mlflow run . -P steps=basic_cleaning
```

Try different hyperparameters for the model:
```bash
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

Run a specific version straight from GitHub:
```bash
mlflow run https://github.com/soniaccodes/Project-Build-an-ML-Pipeline-Starter.git \
  -v 1.0.1 \
  -P hydra_options="etl.sample='sample2.csv'"
```

## Versions

- **v1.0.0** — my first working release. It actually fails when run on a second data sample (`sample2.csv`), because that sample had a few listings with locations outside NYC.
- **v1.0.1** — fixed that by adding a filter that removes listings outside NYC's normal latitude/longitude range. This version runs successfully on the new data.

## Results

- Validation set: R² ≈ 0.55, MAE ≈ $34
- Test set: R² ≈ 0.56–0.57, MAE ≈ $32–34
- After retraining on the new sample (v1.0.1): R² ≈ 0.58, MAE ≈ $32