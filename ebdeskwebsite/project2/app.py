from flask import Flask, request, render_template, url_for, send_from_directory, jsonify
from media_ai import MediaIntelligenceSystem
import os
import io
from werkzeug.utils import secure_filename
import chardet

# For PDF extraction
import PyPDF2
import html2text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'doc', 'docx', 'html', 'htm'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('analysis_outputs', exist_ok=True)

analyzer = MediaIntelligenceSystem(output_dir="analysis_outputs")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file):
    """Extract text from various file formats"""
    filename = file.filename.lower()
    content = file.read()  # Read file content
    file.seek(0)  # Reset file pointer for potential reuse
    
    # PDF extraction
    if filename.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    # HTML extraction
    elif filename.endswith(('.html', '.htm')):
        try:
            # Detect encoding
            detected = chardet.detect(content)
            encoding = detected['encoding'] or 'utf-8'
            html_content = content.decode(encoding, errors='replace')
            
            # Convert HTML to plain text
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            text = converter.handle(html_content)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from HTML: {str(e)}")
    
    # Text files
    elif filename.endswith('.txt'):
        try:
            # Try detecting encoding with chardet
            detected = chardet.detect(content)
            detected_encoding = detected['encoding'] or 'utf-8'
            
            try:
                return content.decode(detected_encoding)
            except UnicodeDecodeError:
                # Try various encodings if the detected one fails
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings_to_try:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                
                # Fall back to replacement mode if all else fails
                return content.decode('utf-8', errors='replace')
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    else:
        raise Exception(f"Unsupported file format: {filename}")

@app.route('/', methods=['GET', 'POST'])
def index():
    report_path = None
    summary = None
    error = None

    if request.method == 'POST':
        # Check if text input was provided
        text = request.form.get('text')
        
        if text and text.strip():
            # Process text input
            result = analyzer.analyze_text_document(text.strip(), title="Text Processing Report")
            report_path = result.get('html_report_path')
            summary = analyzer.generate_text_summary(text)
        else:
            error = "Please enter some text or upload a file."

    return render_template('index.html', report_path=report_path, summary=summary, error=error)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    """Endpoint to extract text from uploaded files"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Extract text from the file
            text = extract_text_from_file(file)
            
            # Save the file for record-keeping (optional)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.seek(0)  # Reset file pointer after extraction
            file.save(filepath)
            
            return jsonify({'text': text})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(app.config['ALLOWED_EXTENSIONS'])})

@app.route('/report/<path:filename>')
def reports(filename):
    return send_from_directory('analysis_outputs', filename)

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)