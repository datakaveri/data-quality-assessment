print("Starting unstructured main.py")  
import logging
from dotenv import load_dotenv
import json, os
import unstructured_metrics.metadata_parser as metadata_parser
from report.aggregate_unstructured import generate_raw_report, generate_final_report
import report.scoring_unstructured as scoring
from report.multifile_average_score import calculate_average_readiness
from report.dataset_clean_name_api import get_dataset_name_from_url
from report.json_writer import write_report_outputs
from report.pdf_writer import generate_pdf_from_json
from unstructured_metrics.llm_api import infer_metadata_roles_openai
from report.post_to_cat_api import update_cat_readiness_score

print("Importing modules completed in main.py")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
logging.info("OpenAI API key loaded successfully.")
elastic_id = os.getenv("ELASTIC_ID")
elastic_pass = os.getenv("ELASTIC_PASS")
logging.info("Elastic credentials loaded successfully.")

def get_output_dir(directory):
    # Use /tmp/outputReports in Lambda, else local outputReports
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return os.path.join("/tmp", os.path.basename(directory))
    else:
        return f"outputReports/{os.path.basename(directory)}"

def log_and_call(func, *args, **kwargs):
    logging.info(f"Calling function: {func.__name__}")
    return func(*args, **kwargs)

def main(directory, folder_key):
    print(f"Running unstructured_main.main on {directory} with folder_key={folder_key}")

    """
    Main function to run the entire data readiness report pipeline for unstructured data.
    - Processes each subdirectory as a dataset, or the root directory if there are no subdirectories.
    - Extracts metadata, infers roles, scores, and generates reports.
    """
    try:
        all_scores = []
        report_names = []

        # Build a list of dataset paths to process
        subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if subdirs:
            dataset_paths = subdirs
        else:
            dataset_paths = [directory]

        for dataset_path in dataset_paths:
            logging.info(f"Processing dataset folder: {dataset_path}")

            # 1. Extract metadata for up to 10 files in this dataset folder
            metadata = log_and_call(metadata_parser.process_folder_to_metadata_json, dataset_path)
            logging.info(f"Extracted metadata for {len(metadata)} files in {dataset_path}")
            print(type(metadata))
            if not metadata:
                logging.error(f"No metadata files found in {dataset_path}")
                continue

            dataset_name = os.path.basename(dataset_path)
            try:
                true_name, uuid = log_and_call(get_dataset_name_from_url, folder_key)
            except Exception:
                true_name = dataset_name
                uuid = None
                logging.info(f"Could not fetch true name for {dataset_path}, using directory name: {true_name}")

            # 2. Use OpenAI to infer roles 
            imputed_roles = log_and_call(infer_metadata_roles_openai, metadata, api_key)
            logging.info(f"Inferred roles for {uuid}: {imputed_roles}")

            # 3. Generate the raw readiness report (based on metadata)
            init_report = log_and_call(generate_raw_report, dataset_path, imputed_roles)

            # 4. Compute the aggregate score (using unstructured scoring)
            final_score = log_and_call(scoring.compute_aggregate_score, init_report)
            final_percentage = final_score.get("total_percentage")
            final_percentage = str(final_percentage) if final_percentage is not None else "unknown"

            # 5. Create a directory to hold all the generated files
            output_dir = get_output_dir(dataset_path)

            # 6. Write the raw and final reports to JSON files
            log_and_call(write_report_outputs, final_score, output_dir, dataset_name, init_report)
            final_report = log_and_call(generate_final_report, f"{output_dir}/{dataset_name}_raw_readiness_report.json")
            with open(f"{output_dir}/{dataset_name}_final_readiness_report.json", "w") as f:
                json.dump(final_report, f, indent=4)
            logging.info(f"Report generated for {dataset_path}")

            # 7. Generate a PDF report
            pdf_output = f"{output_dir}/data_readiness_report.pdf"
            logo_path = "plots/pretty/mahaagx-logo-dark.png"
            sample_size = len(metadata)
            sample = False
            log_and_call(
                generate_pdf_from_json,
                f"{output_dir}/{dataset_name}_final_readiness_report.json",
                pdf_output,
                uuid,
                final_score["total_percentage"],
                output_dir,
                true_name,
                logo_path,
                sample_size,
                sample
            )
            logging.info(f"PDF generated for {dataset_path}")

            all_scores.append(final_score)
            report_names.append(f"{output_dir}/{dataset_name}_raw_readiness_report.json")

            # 8. Update CAT API
            update_cat_readiness_score(uuid, final_percentage, elastic_id, elastic_pass)

        # 9. If there are multiple datasets, generate an average score report
        if len(all_scores) > 1:
            output_dir = get_output_dir(directory)
            raw_avg_report, average_percentage = log_and_call(calculate_average_readiness, report_names)
            with open(f"{output_dir}/average_score_readiness_report.json", "w") as f:
                json.dump(raw_avg_report, f, indent=4)
            final_avg_report = log_and_call(generate_final_report, f"{output_dir}/average_score_readiness_report.json")
            with open(f"{output_dir}/average_score_final_readiness_report.json", "w") as f:
                json.dump(final_avg_report, f, indent=4)
            pdf_output = f"{output_dir}/data_readiness_report.pdf"
            log_and_call(
                generate_pdf_from_json,
                f"{output_dir}/average_score_final_readiness_report.json",
                pdf_output,
                uuid,
                raw_avg_report["total_percentage"],
                output_dir,
                true_name,
                logo_path,
                sample_size,
                average_report=True
            )
            logging.info("Average score report generated for all datasets")
            final_percentage = average_percentage if average_percentage is not None else "unknown"
            update_cat_readiness_score(uuid, final_percentage, elastic_id, elastic_pass)

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()