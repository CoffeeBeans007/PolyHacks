import pdfkit
import os
import img2pdf

# Specify the path to wkhtmltopdf executable (update this path if necessary)
wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

# Set the options for wkhtmltopdf (you can customize this)
options = {
    'page-size': 'A4',
    'orientation': 'Portrait',
    'dpi': 300,
}

# Directory containing your HTML and WebP files
input_directory = r'D:\ssd\Start_Here_Mac.app\Saves\CGsE\Tryts'

# Get a list of HTML and WebP files in the directory
input_files = [f for f in os.listdir(input_directory) if f.endswith(('.html', '.webp'))]

# Sort the input files alphabetically
input_files.sort()

# Initialize PDFKit configuration
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

# Output PDF file name
output_pdf = 'output.pdf'

# Create a list to store file paths
file_paths = []

# Loop through the input files, convert them to PDF, and store their paths
for input_file in input_files:
    file_ext = os.path.splitext(input_file)[1]
    if file_ext == '.html':
        # Convert HTML to PDF
        input_html = os.path.join(input_directory, input_file)
        output_pdf = os.path.splitext(input_file)[0] + '.pdf'
        output_pdf = os.path.join(input_directory, output_pdf)
        
        pdfkit.from_file(input_html, output_pdf, configuration=config, options=options)
        file_paths.append(output_pdf)
    elif file_ext == '.webp':
        # Convert WebP to PDF page
        input_webp = os.path.join(input_directory, input_file)
        output_pdf_page = os.path.splitext(input_file)[0] + '.pdf'
        output_pdf_page = os.path.join(input_directory, output_pdf_page)
        
        with open(output_pdf_page, 'wb') as pdf_page:
            pdf_page.write(img2pdf.convert(input_webp))
        
        file_paths.append(output_pdf_page)

# Combine all PDF pages into a single PDF
with open(os.path.join(input_directory, 'combined.pdf'), 'wb') as combined_pdf:
    combined_pdf.write(img2pdf.convert(file_paths))

# Clean up - optional
# If you want to delete the individual PDF pages, you can do so after combining them.
# for pdf_page in file_paths:
#     os.remove(pdf_page)

print("PDF files generated successfully!")
