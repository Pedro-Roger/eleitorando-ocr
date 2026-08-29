import sys, os, json, base64, io, re, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

# carrega igual ao main.py
import main as m

img_path = sys.argv[1]
img = Image.open(img_path)
if img.mode != 'RGB':
    img = img.convert('RGB')
# limita tamanho para não estourar payload
max_side = 1600
w, h = img.size
scale = max_side / max(w, h)
if scale < 1:
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

buf = io.BytesIO()
img.save(buf, format='JPEG', quality=90)
b64 = base64.b64encode(buf.getvalue()).decode()

prompt = (
    "Esta é uma foto de uma caderneta de campo com UMA OU VÁRIAS fichas de eleitores "
    "escritas à mão. Cada ficha tem: Nome, Endereço, Bairro, Telefone, Nº Título, Seção, Zona.\n"
    "Leia com atenção (letra de mão, inclua rabiscos e palavras entre parênteses como (FILHO)) "
    "e extraia TODAS as fichas na ordem em que aparecem.\n"
    "Responda APENAS um JSON válido, sem texto extra, no formato:\n"
    '{"voters": [{"nome": "", "telefone": "somente dígitos", "titleNumber": "somente dígitos", '
    '"secao": "número", "zona": "número", "bairro": ""}]}\n'
    "Telefone: 10-11 dígitos com DDD. Título: 10-12 dígitos, sem pontos. "
    "Se um campo estiver ilegível, devolva string vazia. Não invente valores."
)

body = json.dumps({
    'model': m.OPENROUTER_MODEL,
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
        ],
    }],
    'max_tokens': 3000,
}).encode()

req = urllib.request.Request(
    'https://openrouter.ai/api/v1/chat/completions',
    data=body,
    headers={'Authorization': f'Bearer {m.OPENROUTER_API_KEY}', 'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read())
content = data['choices'][0]['message']['content']
mm = re.search(r'\{.*\}', content, re.DOTALL)
parsed = json.loads(mm.group(0))
print(json.dumps(parsed, ensure_ascii=False, indent=2))
