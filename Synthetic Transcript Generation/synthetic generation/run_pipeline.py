import os
import sys
import subprocess
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CCDA_DIR = os.path.join(SCRIPT_DIR, "..", "Synthea CCDAs")

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed at step: {description}")
        print(f"Command run: {' '.join(command)}")
        sys.exit(1)

def execute_pipeline(ccda_path, symptoms_list=None):
    """
    Programmatic interface to run the pipeline.
    You can import this function in a Jupyter Notebook or another script:
    
    from run_pipeline import execute_pipeline
    execute_pipeline("../Synthea CCDAs/Adela471...xml", ["Purple toes", "Loss of taste"])
    """
    if symptoms_list is None:
        symptoms_list = []
        
    print(f"\nStarting Pipeline for: {os.path.basename(ccda_path)}")

    # Step 1: Parse CCDA to JSON
    run_step(
        [sys.executable, os.path.join(SCRIPT_DIR, "ccda_to_ground_truth.py"), ccda_path],
        "Step 1: Parse C-CDA to Ground Truth JSON"
    )

    # Step 2: Inject Symptoms
    ccda_filename = os.path.basename(ccda_path)
    base_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    modified_json_name = ccda_filename.replace(".xml", "_modified_ground_truth.json")
    
    base_json_path = os.path.join(SCRIPT_DIR, base_json_name)
    modified_json_path = os.path.join(SCRIPT_DIR, modified_json_name)
    
    if symptoms_list:
        inject_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "inject_symptoms.py"),
            "--input", base_json_path,
            "--output", modified_json_path
        ]
        # Instead of resetting, we just inject directly from the fresh base JSON
        
        # Now inject the new ones
        for symp in symptoms_list:
            inject_cmd.extend(["--symptom", symp])
        run_step(
            inject_cmd,
            "Step 2: Injecting Novel Symptoms"
        )
    else:
        print(f"\n{'='*60}")
        print("  Step 2: Skipped (no symptoms provided to inject)")
        print(f"{'='*60}")

    # Step 3: Generate the synthetic transcript
    run_step(
        [sys.executable, os.path.join(SCRIPT_DIR, "generate_synthetic_transcript.py"), "--ccda", ccda_path],
        "Step 3: Generate Synthetic Transcript via LLM"
    )
    
    # Step 4: Open Viewer
    viewer_path = os.path.join(SCRIPT_DIR, "..", "..", "Dashboards + Annotater", "ccda_comparison_viewer.py")
    
    gt_path = modified_json_path if symptoms_list else base_json_path
    if not os.path.exists(gt_path):
        gt_path = base_json_path
        
    run_step(
        [sys.executable, viewer_path, gt_path],
        "Step 4: Launching Comparison Viewer"
    )
    
    print("\n✓ Pipeline completed successfully!")


def main_interactive():
    print("======================================================")
    print("      Synthetic Transcript Generation Pipeline")
    print("======================================================")

    if not os.path.exists(CCDA_DIR):
        print(f"ERROR: CCDA directory not found: {CCDA_DIR}")
        return

    xml_files = glob.glob(os.path.join(CCDA_DIR, "*.xml"))
    if not xml_files:
        print(f"No XML files found in {CCDA_DIR}")
        return

    print("\nAvailable Synthea CCDA files:")
    for i, file_path in enumerate(xml_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    # Select CCDA
    while True:
        try:
            choice = input("\nSelect a CCDA file by number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return
            idx = int(choice) - 1
            if 0 <= idx < len(xml_files):
                selected_ccda = xml_files[idx]
                break
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Prompt for symptoms
    print(f"\nSelected: {os.path.basename(selected_ccda)}")
    symptoms_input = input("Enter novel symptoms to inject (comma-separated), or press Enter to skip: ").strip()
    
    symptoms_list = []
    if symptoms_input:
        symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    # Execute the actual pipeline
    execute_pipeline(selected_ccda, symptoms_list)

if __name__ == "__main__":
    main_interactive()
