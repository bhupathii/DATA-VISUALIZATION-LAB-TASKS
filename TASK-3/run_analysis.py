import subprocess
import sys

print("=== AGRICULTURE CROP YIELD BIVARIATE ANALYSIS ===")
print("Running separate analyses for different data types...\n")

# Run Continuous vs Continuous Analysis
print("1. Running Continuous vs Continuous Analysis...")
try:
    subprocess.run([sys.executable, 'continuous_vs_continuous.py'], check=True)
    print("✅ Continuous vs Continuous analysis completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error in continuous vs continuous analysis: {e}")

print("\n" + "="*50 + "\n")

# Run Categorical vs Continuous Analysis
print("2. Running Categorical vs Continuous Analysis...")
try:
    subprocess.run([sys.executable, 'categorical_vs_continuous.py'], check=True)
    print("✅ Categorical vs Continuous analysis completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error in categorical vs continuous analysis: {e}")

print("\n" + "="*50)
print("=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("- continuous_vs_continuous.png")
print("- categorical_vs_continuous.png")
print("\nCheck the generated PNG files for visualizations!") 