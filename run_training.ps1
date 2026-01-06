# PowerShell script to run training with proper PYTHONPATH
# Usage: .\run_training.ps1 --config YourProjectName/configs/your_project.yaml

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ConfigArgs
)

# Set PYTHONPATH to current directory
$env:PYTHONPATH = "."

# Run the training script with all arguments
python Regression/mainfittest.py $ConfigArgs

