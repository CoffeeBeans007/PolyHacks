import pdfkit
import os

pdfkit_options = {
    'page-size': 'A4',
    'dpi': 300,
    'debug': 'true',  # Add this line for debug output
}

# Specify the path to wkhtmltopdf executable (update this path if necessary)
wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

# Set the options for wkhtmltopdf (you can customize this)
options = {
    'page-size': 'A4',
    'orientation': 'Portrait',
    'dpi': 300,
}

# Directory containing your HTML files
html_directory = r'D:\ssd\Start_Here_Mac.app\Saves\CGsE\Tryts'

# Get a list of HTML files in the directory
html_files = [f for f in os.listdir(html_directory)]
print(html_files)
# Sort the HTML files alphabetically
html_files.sort()

# Initialize PDFKit configuration
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

# Output PDF file name
output_pdf = 'output.pdf'

# Create a list to store file paths
file_paths = []

# Loop through the HTML files, convert them to PDF, and store their paths
for html_file in html_files:
    input_html = os.path.join(html_directory, html_file)
    output_pdf = os.path.splitext(html_file)[0] + '.pdf'
    output_pdf = os.path.join(html_directory, output_pdf)
    
    pdfkit.from_file(input_html, output_pdf, configuration=config, options=options)
    file_paths.append(output_pdf)

# Combine all PDF files into a single PDF (optional)
# You can use a library like PyPDF2 to combine PDF files.

# Clean up - optional
# If you want to delete the individual PDF files, you can do so after combining them.
# for pdf_file in file_paths:
#     os.remove(pdf_file)

print("PDF files generated successfully!")
