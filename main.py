from data_preprocessing import load_and_preprocess
from model_training import run_training
import subprocess
def main():
    print("=" * 60)
    print("Fatigue Life Prediction of Stainless Steels")
    print("Knowledge-Driven Feature Engineering + ML")
    print("=" * 60)

    # Step 1: Data preprocessing
    print("\n[Step 1] Data Preprocessing")
    print("-" * 40)
    df, le = load_and_preprocess('fatigue_dataset.xlsx')

    # Step 2: Model training
    print("\n[Step 2] Model Training")
    print("-" * 40)
    best_models, results_df = run_training(df)

    # Step 3: Figure generation
    print("\n[Step 3] Figure Generation")
    print("-" * 40)
    subprocess.run(['python', 'plot_Fig2_correlation.py'])
    subprocess.run(['python', 'plot_Fig3_predictions.py'])

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Outputs: model_results.csv, predictions.npz,")
    print("Fig2_correlation.tiff, Fig3_predictions.tiff")
    print("=" * 60)

if __name__ == '__main__':
    main()