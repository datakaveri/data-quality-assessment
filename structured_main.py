print("Starting structured main.py")  
import logging
from dotenv import load_dotenv
import json, os
import report.input_handler as input_handler
from report.aggregate_structured import generate_raw_report, generate_final_report
import report.scoring_structured as scoring
from report.multifile_average_score import calculate_average_readiness
from report.dataset_clean_name_api import get_uuid_from_dataset_name, get_dataset_name_from_url
from report.json_writer import write_report_outputs
from report.pdf_writer import generate_pdf_from_json
from structured_metrics.llm_api import infer_column_roles_openai
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
    """
    Main function to run the entire data readiness report pipeline.

    This function will:

    1. Ask the user for a directory containing data files.
    2. Load all the data files in the directory.
    3. Make an API call to OpenAI to infer column roles.
    4. Run the raw readiness report for each file.
    5. Compute the aggregate score for each file.
    6. Write the raw and final reports to JSON files.
    7. Generate a PDF report for each file.
    8. If there are multiple files, generate a report with the average score across all the files.
    """
    # directory = input("Enter the directory containing data files: ")

    try:
        data = log_and_call(input_handler.load_data_from_directory, directory)
        all_scores = []
        logging.info(f"Loaded {len(data)} files from {directory}")
        if not data:
            logging.error("No data files found in the specified directory.")
            return
        for df, file_path, sample in data:
            try:
                # Get the dataset name from the file path, strip special characters
                dataset_name = os.path.splitext(os.path.basename(file_path))[0].replace('%20', ' ').replace('%21', '!').replace('%22', '"').replace('%23', '#').replace('%24', '$').replace('%25', '%').replace('%26', '&').replace('%27', "'").replace('%28', '(').replace('%29', ')').replace('%2A', '*').replace('%2B', '+').replace('%2C', ',').replace('%2D', '-').replace('%2E', '.').replace('%2F', '/').replace('%3A', ':').replace('%3B', ';').replace('%3C', '<').replace('%3D', '=').replace('%3E', '>').replace('%3F', '?').replace('%40', '@').replace('[', '(').replace(']', ')')
                try:
                    true_name, uuid = log_and_call(get_dataset_name_from_url, folder_key)
                except Exception:
                    true_name = os.path.basename(directory)
                    logging.info(f"Could not fetch true name for {file_path}, using directory name: {true_name}")

                sample_size = len(df)
                logging.info(f"Sample size for {dataset_name}: {sample_size} rows")


                file_path = os.path.dirname(file_path)

                # Use OpenAI to infer column roles
                imputed_columns = log_and_call(infer_column_roles_openai, df, api_key)
                logging.info(f"Inferred column roles for {uuid}: {imputed_columns}")

                # Generate the raw readiness report
                init_report = log_and_call(generate_raw_report, df, file_path, imputed_columns)
                
                # Compute the aggregate score
                final_score = log_and_call(scoring.compute_aggregate_score, init_report, df)
                final_percentage = final_score.get("total_percentage")
                final_percentage = str(final_percentage) if final_percentage is not None else "unknown"

                # Create a directory to hold all the generated files
                output_dir = get_output_dir(directory)

                # Write the raw and final reports to JSON files
                log_and_call(write_report_outputs, final_score, output_dir, dataset_name, init_report)
                final_report = log_and_call(generate_final_report, f"{output_dir}/{dataset_name}_raw_readiness_report.json")
                with open(f"{output_dir}/{dataset_name}_final_readiness_report.json", "w") as f:
                    json.dump(final_report, f, indent=4)
                logging.info(f"Report generated for {file_path}")
                
                # Generate a PDF report
                pdf_output = f"{output_dir}/data_readiness_report.pdf"
                logo_path = "plots/pretty/MahaAgriLogo.png"
                log_and_call(generate_pdf_from_json, f"{output_dir}/{dataset_name}_final_readiness_report.json", pdf_output, uuid, final_score["total_percentage"], output_dir, true_name, logo_path, sample_size, sample)
                logging.info(f"PDF generated for {file_path}")
                
                all_scores.append(final_score)
                report_names = [f"{output_dir}/{dataset_name}_raw_readiness_report.json" for _, file_path, _ in data]

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                logging.info(f"Skipping {dataset_name}")
        
        # If there are multiple files, generate a report with the average score across all the files
        if len(all_scores) > 1:
            output_dir = get_output_dir(directory)
            raw_avg_report, average_percentage = log_and_call(calculate_average_readiness, report_names)

            with open(f"{output_dir}/average_score_readiness_report.json", "w") as f:
                json.dump(raw_avg_report, f, indent=4)

            final_avg_report = log_and_call(generate_final_report, f"{output_dir}/average_score_readiness_report.json")
            with open(f"{output_dir}/average_score_final_readiness_report.json", "w") as f:
                json.dump(final_avg_report, f, indent=4)

            pdf_output = f"{output_dir}/data_readiness_report.pdf"
            logo_path = "plots/MahaAgriLogo.png"  # Set this to None if not needed
            log_and_call(generate_pdf_from_json, f"{output_dir}/average_score_final_readiness_report.json", pdf_output, uuid, raw_avg_report["total_percentage"], output_dir, true_name, logo_path, sample_size, average_report=True)
            logging.info("Average score report generated for all datasets")
            final_percentage = average_percentage if average_percentage is not None else "unknown"
        update_cat_readiness_score(uuid, final_percentage, elastic_id, elastic_pass)
        return
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()

