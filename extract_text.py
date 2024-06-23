import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = 'data/course_details.pdf'
course_text = extract_text_from_pdf(pdf_path)
with open('data/course_details.txt', 'w') as f:
    f.write(course_text)
