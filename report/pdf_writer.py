from fpdf import FPDF
import json
import datetime

class PDFReport(FPDF):
    @staticmethod
    def sanitize_text(text):
        # Replace common Unicode dashes and quotes with ASCII equivalents
        if not isinstance(text, str):
            return text
        return (
            text.replace('\u2013', '-')
                .replace('\u2014', '-') 
                .replace('\u2018', "'")
                .replace('\u2019', "'")
                .replace('\u201c', '"')
                .replace('\u201d', '"')
        )
    def __init__(self, dataset_name, total_percentage, directory, true_name, sample_size, logo_path=None, sample=False, average_report=False):
        """
        Constructor for PDFReport

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        total_score : float
            Total score of the dataset
        logo_path : str, optional
            Path to the logo to be displayed on the top left of the report, by default None
        """
        super().__init__()
        self.sample = sample
        self.average_report = average_report
        self.directory = directory
        self.sample_size = sample_size
        self.true_name = self.sanitize_text(true_name)
        # self.true_name = "Cropsap ETL and Near ETL"
        if self.average_report:
            self.dataset_name = f"{self.true_name} - Average Report"
        elif self.sample:
            self.dataset_name = f"{self.true_name} (Sampled - {self.sample_size} rows)"
        else:
            self.dataset_name = self.true_name
        self.percent_score = total_percentage
        self.logo_path = logo_path
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Helvetica", size=10)

    def header(self):
        if self.logo_path:
            # self.image(self.logo_path, 10, 5, 25)  # TGDEX Logo at top-left, width = 25
            self.image(self.logo_path, 8, 8, 50)  # MahaAgri Logo at top-left, width = 50

        self.set_font("Helvetica", 'B', 18)
        self.set_xy(10, 10)
        self.cell(0, 10, 'Data Readiness Report', ln=True, align='C')

        self.set_font("Helvetica", '', 10)
        self.set_xy(9, 30)
        self.cell(0, 10, f"Dataset: {self.dataset_name}", ln=True, align='L')
        # self.cell(0, 10, f"Dataset: {self.dataset_name}", ln=True, align='L')

        self.set_font("Helvetica", '', 10)
        self.set_xy(9, 35)
        self.cell(0, 10, f"Report Generated On: {datetime.datetime.now().strftime('%d-%b-%Y')}", ln=True, align='L')

        # self.set_font("Helvetica", 'B', 12)
        # self.set_xy(160, 30)
        # self.cell(0, 10, f"Score: {self.total_score:.2f} / {self.total_weights:.2f}", ln=True, align='L')
        
        self.set_font("Helvetica", 'B', 10)
        self.set_xy(170, 35)
        self.cell(0, 10, f"Percentage: {self.percent_score:.2f}%", ln=True, align='L')


        # self.ln(5)

    def render_table(self, data):
        """
        Render the table of data readiness checks from the given data.

        Parameters
        ----------
        data : list
            List of sections, each containing a bucket name, weight, and a list of tests.
            Each test is a dictionary with keys 'id', 'title', 'note', 'score', and 'max_score'.
        """
        col_widths = [15, 40, 115, 13, 12]
        headers = [' Test ID', '       Test Description', '                                              Summary of Findings', ' Score', ' Max']

        # Table headers
        self.set_font("Helvetica", 'B', 9)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, border=1)
        self.ln()

        self.set_font("Helvetica", '', 9)
        # Filter out sections where all tests have max_score == 0
        filtered_data = []
        for section in data:
            # Remove tests with max_score == 0 if there are other tests in the section
            filtered_tests = [test for test in section['tests'] if test['max_score'] != 0]
            if not filtered_tests and len(section['tests']) == 1:
                continue  # Skip section if only one test and its max_score == 0
            elif filtered_tests:
            # Keep section with filtered tests
                section_copy = section.copy()
                section_copy['tests'] = filtered_tests
                filtered_data.append(section_copy)
            else:
            # If all tests have max_score == 0 but more than one test, remove those tests
                continue
        
        section_num = 1
        for section in filtered_data:
            if any(test['max_score'] != 0 for test in section['tests']):
            # Bucket heading
                self.set_fill_color(200, 240, 200)  # Light green
                self.set_text_color(0, 0, 0)  # Black text
                self.set_font("Helvetica", 'B', 10)
                self.cell(sum(col_widths), 8, f"{section['bucket']}", ln=True, fill=True, border=1)
                self.set_fill_color(255, 255, 255)  # White background
                self.set_text_color(0, 0, 0)  # Black text
                self.set_font("Helvetica", '', 9)

            test_num = 1
            for test in section['tests']:
                # Save starting X and Y
                x_start = self.get_x()
                y_start = self.get_y()

                # Calculate height required for the 'note' cell
                note_text = test['note']
                note_width = col_widths[2]
                note_line_height = 8
                note_lines = self.multi_cell(note_width, note_line_height, note_text, border=0, split_only=True)
                note_height = note_line_height * len(note_lines)

                # Draw cell 1: test number
                self.set_xy(x_start, y_start)
                self.multi_cell(col_widths[0], note_height, f"{section_num}.{test_num}", border=1)

                # Draw cell 2: title
                self.set_xy(x_start + col_widths[0], y_start)
                self.multi_cell(col_widths[1], note_height, test['title'], border=1)

                # Draw cell 3: note (multi-line content)
                self.set_xy(x_start + col_widths[0] + col_widths[1], y_start)
                self.multi_cell(col_widths[2], note_line_height, note_text, border=1)

                # Draw cell 4: score
                self.set_xy(x_start + col_widths[0] + col_widths[1] + col_widths[2], y_start)
                self.cell(col_widths[3], note_height, f"{test['score']:.2f}", border=1, align='C', fill=True)

                # Draw cell 5: max score
                self.set_xy(x_start + col_widths[0] + col_widths[1] + col_widths[2] + col_widths[3], y_start)
                self.cell(col_widths[4], note_height, f"{test['max_score']:.0f}", border=1, align='C', fill=True)

                # Move to the next line
                self.set_y(y_start + note_height)

                test_num += 1
        
            # Total score for each bucket
            self.set_font("Helvetica", 'B', 10)
            self.set_fill_color(200, 211, 211)  # Light gray
            self.cell(col_widths[0]+col_widths[1]+col_widths[2], 8, f"Subtotal", border=1, fill=True)
            self.cell(col_widths[3], 8, f"{sum(test['score'] for test in section['tests']):.2f}", border=1, align='C', fill=True)
            self.cell(col_widths[4], 8, f"{sum(test['max_score'] for test in section['tests']):.0f}", border=1, align='C', fill=True)
            self.ln()
            section_num+=1 

def generate_pdf_from_json(json_path, output_path, dataset_name, total_percentage, directory, true_name, logo_path=None, sample_size=None, sample=False, average_report=False):
    """
    Generates a PDF report from a JSON file containing readiness data.

    Parameters
    ----------
    json_path : str
        The path to the JSON file containing the readiness data.
    output_path : str
        The path where the generated PDF report will be saved.
    dataset_name : str
        The name of the dataset for which the report is generated.
    total_score : float
        The overall readiness score of the dataset.
    logo_path : str, optional
        The path to the logo image to be included in the report header (default is None).

    Returns
    -------
    None
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    pdf = PDFReport(dataset_name, total_percentage, directory, true_name, sample_size, logo_path, sample, average_report)
    pdf.render_table(data)
    # pdf.output(dest=output_path).encode('utf-8','ignore')
    pdf.output(output_path)
