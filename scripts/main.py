import subprocess

def run_step(name, command):
    print(f"\n Running: {name}")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f" Y {name} Successful!")
        
    else:
        print(f" N {name} Fail, returncode: {result.returncode}")
        exit(result.returncode)

if __name__ == "__main__":
    run_step("Step 1 - Feature Builder", "python scripts/features_builder.py")
    run_step("Step 2 - Model Builder", "python scripts/model_builder.py")
    run_step("Step 3 - Model Evaluation", "python scripts/model_evaluation.py")

    print("\n Successful!")
