"""
Training entrypoint script for Customer Segmentation and Churn Prediction.
Run this script to re-train all models and regenerate model artifacts.
"""
import sys
import time
from src.model_pipeline import ModelPipeline

def main():
    print("=" * 60)
    print("  Customer Segmentation & Churn Prediction Training Pipeline  ")
    print("=" * 60)
    start_time = time.time()
    pipeline = ModelPipeline(data_path="TelcoCustomerChurn.csv", output_dir="models")
    metadata = pipeline.run_training_pipeline()
    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline completed in {elapsed:.2f} seconds!")
    print("\nModel Evaluation Leaderboard:")
    for res in metadata["evaluation_results"]:
        print(f" - {res['Model']:<28}: Accuracy={res['Accuracy']:.4f}, F1={res['F1_Score']:.4f}, ROC-AUC={res['ROC_AUC']}")

if __name__ == "__main__":
    main()
