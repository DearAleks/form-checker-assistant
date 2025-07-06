from flask import Flask, request, jsonify, render_template_string
import os
from dotenv import load_dotenv
from vector_db import VectorDatabase
from document_processor import DocumentProcessor
import openai
import tempfile
import traceback

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Создаем папки если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.getenv('REFERENCE_DOCS_FOLDER', './reference_docs'), exist_ok=True)

# Инициализация компонентов
try:
    vector_db = VectorDatabase()
    doc_processor = DocumentProcessor()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Проверяем, что API ключ установлен
    if not openai.api_key:
        print("⚠️  ВНИМАНИЕ: OPENAI_API_KEY не установлен в переменных окружения!")
        
except Exception as e:
    print(f"Ошибка при инициализации компонентов: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mikrokvalifikatsiooni õppe õppekava vormi hindaja</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Mulish, sans-serif; 
                margin: 40px; 
                background-color: #f4f4f4;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background-color: #cdd6ae;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .upload-area { 
                border: 2px dashed #588157; 
                padding: 30px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 8px;
                background-color: #cdd6ae;
            }
            .upload-area:hover {
                background-color: #f4f4f4;
            }
            button { 
                background-color: #588157; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover { 
                background-color: #112549; 
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                background-color: #f4f4f4; 
                border-radius: 4px;
                border-left: 4px solid #588157;
            }
            .error {
                background-color: #f4f4f4;
                border-left: 4px solid #f57e30;
                color: #721c24;
            }
            .loading {
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f57e30;
                border-top: 4px solid #588157;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mikrokvalifikatsiooni õppe õppekava vormi hindaja</h1>
            <p>TERE TULEMAST! Olen sinu digitaalne nõustaja mikrokvalifikatsiooni õppe õppekava vormi täitmisel. Kontrollin, kas mikrokvalifikatsiooni õppe õppekava vorm on täidetud korrektselt. Alustuseks palun lae üles oma mikrokvalifikatsiooni õppekava vorm.</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="fileInput" accept=".pdf,.txt" required>
                    <p>Vali PDF või tekstifail (maksimum 16MB)</p>
                </div>
                <button type="submit" id="submitBtn">Alusta vormi hindamist</button>
            </form>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    resultDiv.innerHTML = '<div class="error">Palun valige fail!</div>';
                    resultDiv.style.display = 'block';
                    return;
                }
                
                formData.append('file', fileInput.files[0]);
                
                // Показываем загрузку
                submitBtn.disabled = true;
                submitBtn.textContent = 'Hindan...';
                resultDiv.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Hindan vormi, palun oodake...</p>
                    </div>
                `;
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/check_form', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const contentType = response.headers.get('content-type');
                    
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error('Server tagastas vale vormingu. Võib-olla on server häiritud.');
                    }
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3>Hindamise tulemus:</h3>
                            <div style="white-space: pre-wrap; line-height: 1.6;">${result.analysis}</div>
                            <hr>
                            <small>Analüüsis kasutati ${result.similar_docs_count} viitedokumenti</small>
                        `;
                        resultDiv.className = 'result';
                    } else {
                        resultDiv.innerHTML = `<div class="error">Viga: ${result.error}</div>`;
                        resultDiv.className = 'result error';
                    }
                    
                } catch (error) {
                    console.error('Detailed error:', error);
                    resultDiv.innerHTML = `<div class="error">Viga vormi hindamisel: ${error.message}</div>`;
                    resultDiv.className = 'result error';
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Alusta vormi hindamist';
                }
            });
        </script>
    </body>
    </html>
    ''')
@app.route('/health')
def health():
    """"Äpi tervise kontroll"""
    try:
        stats = vector_db.get_stats()
        return jsonify({
            'status': 'healthy',
            'database_ready': stats['total_documents'] > 0,
            'documents_count': stats['total_documents'],
            'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'database_ready': False,
            'documents_count': 0,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
        }), 500
    
@app.route('/check_form', methods=['POST'])
def check_form():
    """Alustan vormi hindamist"""
    try:
        # Проверяем наличие файла
        if 'file' not in request.files:
            return jsonify({'error': 'Faili ei leitud'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Faili ei valitud'}), 400
        
        # Проверяем расширение файла
        allowed_extensions = {'.pdf', '.txt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Lubatud on ainult PDF ja TXT failid'}), 400
        
        # Проверяем API ключ
        if not openai.api_key:
            return jsonify({'error': 'OpenAI API võti ei ole seadistatud'}), 500
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # Извлекаем текст из файла
            if file_ext == '.pdf':
                form_text = doc_processor.extract_text_from_pdf(tmp_file_path)
            else:
                form_text = doc_processor.extract_text_from_txt(tmp_file_path)
            
            # Проверяем, что текст извлечен
            if not form_text or len(form_text.strip()) < 10:
                return jsonify({'error': 'Teksti failist eraldamine ei õnnestunud või fail on liiga lühike'}), 400
            
            print(f"Извлечено текста: {len(form_text)} символов")
            
            # Поиск похожих документов
            similar_docs = vector_db.search_similar(form_text, n_results=3)
            
            # Извлекаем текст из результатов поиска
            if similar_docs:
                context_texts = [doc['text'] for doc in similar_docs]
                context = "\n\n".join(context_texts)
                similar_docs_count = len(similar_docs)
            else:
                context = "Viitedokumente ei leitud."
                similar_docs_count = 0
            
            # Подготавливаем контекст для OpenAI
            context = "\n\n".join([doc['text'] for doc in similar_docs])
            
            # Создаем промпт
            prompt = f"""
Sa oled vormide analüüsi ja kontrolli ekspert. Analüüsi üleslaetud vormi vastavust viitedokumentides toodud standarditele.

VIITEDOKUMENDID:
{context}

KONTROLLITAV VORM:
{form_text}

ÜLESANNE:
Tee vormi põhjalik analüüs ja esita struktureeritud aruanne:

1. VÄLJADE ANALÜÜS
Korrektselt täidetud väljad:
- Nimekiri õigesti täidetud väli(de)st

Valesti täidetud või puuduvad väljad:
- Probleemsete väljade loetelu

2. KONKREETSED VEAD
- Konkreetsed vead koos viitega vastavale referentsdokumendile

3. STANDARDITEGA VASTAVUS
- Kas vorm vastab viitedokumentide nõuetele
- Vormindusvead
- Kohustuslike elementide puudumine

4. SOOVITUSED
- Konkreetsed parandusetapid
- Paranduste prioriteetsus
- Täiendavad nõuanded

5. LÕPLIK HINNANG
- Täitmise korrektsuse protsent (0-100%)
- Üldstaatus: AKTSEPTEERITUD / VAJAB PARANDUSI / TAGASI LÜKATUD

Vasta eesti keeles, ole konkreetne ja professionaalne. 
Lisasoovitus: Kustutada vormis esinevad kaldkirjas kommentaarid (nt „kui õppekava läbiviimisesse on kaasatud…") kui need on alles jäänud täidetud vormi.
"""
            
            # Запрос к OpenAI
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Oled dokumentide ja vormide kontrollimise ekspert ning analüüsid esitatud materjale põhjalikult."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            
            return jsonify({
                'analysis': analysis,
                'similar_docs_count': similar_docs_count
            })
            
        except Exception as e:
            return jsonify({'error': f'Viga OpenAI API kutsumisel: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return jsonify({'error': 'Viga vormi töötlemisel'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)