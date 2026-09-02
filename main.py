import os
import re
import time
import base64
import asyncio
import unicodedata
import urllib.request
import urllib.error
import json as _json
import uuid as _uuid
import numpy as np
from dataclasses import dataclass, asdict
from fastapi import FastAPI, UploadFile, File
import easyocr
import psycopg2
import psycopg2.extras
from PIL import Image, ImageOps
import io

# carrega OPENROUTER_API_KEY do ../api/.env (mesma origem do backend Node)
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', 'api', '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env_done = None

app = FastAPI()

reader = easyocr.Reader(['pt', 'en'], gpu=False, verbose=False)


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


NOISE_RE = re.compile(
    r'(REPUBLICA|FEDERAT|BRASIL|TITULO|ELEITORAL|ELEITOS|JUSTICA|CODIGO|VALIDACAO|'
    r'EMISSAO|impresso|autenticidade|Tribunal|tse\.jus|Orientacoes|apto[sz]?\b|biometria|'
    r'FILIA|JUIZ|VALIDO|marca|eleitor/eleitora|NASCIMENTO|INSCRIC|ZONA|SECAO|MUNICIPIO|'
    r'NO[MHVK]E\W*DO\W*ELEITOR|KCVE)',
    re.IGNORECASE,
)

# Rótulos tolerantes a erros de OCR (NOHE/NOVE/NOME, Nscuco/INSCRICAO, Zov/ZONA, SEc4o/SECAO)
LABEL_NOME = r'N[O0][MHVK]E\W*DO\W*ELEITORA?'
LABEL_INSC = r'\b(?:INS|NS[CÇK]|NSR)\w*'
LABEL_ZONA = r'\bZO\w{0,2}\b'
LABEL_SECAO = r'\bSEC[A4@O0]O\b|\bSE[CÇ][A4@]O?\b'


def is_noise(line_norm):
    if not line_norm or len(line_norm) < 2:
        return True
    if NOISE_RE.search(line_norm):
        return True
    if re.fullmatch(r'[\d\s/|]+', line_norm):
        return True
    return False


def extract_fields(text):
    result = {'nome': '', 'dataNascimento': '', 'gender': '', 'age': '', 'titleNumber': '', 'zona': '', 'secao': '', 'municipio': '', 'uf': ''}
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    norm = [strip_accents(l).upper() for l in lines]

    # --- passada 1: rótulo e valor na MESMA linha ---
    for i, l in enumerate(norm):
        if not result['nome']:
            m = re.search(LABEL_NOME + r'\W*(.+)', l)
            if m and m.group(1).strip() and not is_noise(strip_accents(m.group(1)).upper()):
                result['nome'] = lines[i][len(lines[i]) - len(m.group(1)):].strip()
        if not result['titleNumber']:
            m = re.search(LABEL_INSC + r'\W*(\d[\d\s.]{9,16}\d)', l)
            if m:
                digits = re.sub(r'\D', '', m.group(1))
                if 10 <= len(digits) <= 14:
                    result['titleNumber'] = digits
        if not result['zona']:
            m = re.search(LABEL_ZONA + r'\W*(\d{1,4})', l)
            if m:
                result['zona'] = m.group(1)
        if not result['secao']:
            m = re.search(r'(?:' + LABEL_SECAO + r')\W*(\d{1,4})', l)
            if m:
                result['secao'] = m.group(1)
        if not result['dataNascimento']:
            m = re.search(r'(\d{2})/(\d{2})/(\d{4})', l)
            if m:
                result['dataNascimento'] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        if not result['municipio']:
            # separador tolerante: " / " ou "/" ou "I" colado (OCR come a barra)
            m = re.search(r'^([A-ZÀ-ÿ][A-ZÀ-ÿ\s]+?)[\'/I]\s*(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\s*$', l)
            # município no documento é CAIXA ALTA — rejeita linhas com
            # minúsculas misturadas (nomes de filiação type "ESPIRITo SanTo")
            letters = [c for c in lines[i] if c.isalpha()]
            upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
            if m and (not letters or upper_ratio >= 0.8):
                raw_city = re.sub(r"[^A-Za-zÀ-ÿ\s]", "", m.group(1)).strip()
                result['municipio'] = raw_city.title()
                result['uf'] = m.group(2)

    def find_label(pattern, start=0):
        for i in range(start, len(norm)):
            if re.search(pattern, norm[i]):
                return i
        return -1

    def find_digit_line(start, end, max_digits=4):
        for j in range(start, min(end, len(lines))):
            if re.fullmatch(r'\d{1,%d}' % max_digits, norm[j].strip()):
                return j
        return -1

    def find_title_line(start, end):
        for j in range(start, min(end, len(lines))):
            digits = re.sub(r'\D', '', norm[j])
            if 10 <= len(digits) <= 14:
                return j
        return -1

    # --- passada 2: título/zona/seção em linhas separadas dos rótulos ---
    # caso A: rótulos na MESMA linha horizontal ("INSCRIÇÃO ... ZONA ... SEÇÃO")
    # e valores na linha seguinte ("num zona seção") — layout do título impresso
    if not (result['titleNumber'] and result['zona'] and result['secao']):
        for i, l in enumerate(norm):
            has_ins = re.search(LABEL_INSC, l) and not re.search(r'\d', l)
            has_zona = re.search(r'\bZON[AEOU]?\b|\bZO[NVU]\b', l)
            has_secao = re.search(LABEL_SECAO, l)
            if not (has_ins and has_zona and has_secao):
                continue
            # procura a linha de valores nas próximas 3 linhas
            for j in range(i + 1, min(i + 4, len(lines))):
                nums = re.findall(r'\d[\d\s.,]*\d|\d', strip_accents(lines[j]))
                if len(nums) < 2:
                    continue
                groups = [re.sub(r'\D', '', g) for g in nums]
                groups = [g for g in groups if g]
                # título: 10-14 dígitos · zona: 1-3 · seção: 3-4
                for g in groups:
                    if 10 <= len(g) <= 14 and not result['titleNumber']:
                        result['titleNumber'] = g
                    elif 1 <= len(g) <= 3 and not result['zona']:
                        result['zona'] = g
                    elif 3 <= len(g) <= 4 and not result['secao']:
                        result['secao'] = g
                if result['titleNumber'] and (result['zona'] or result['secao']):
                    break
            break

    idx_ins = find_label(LABEL_INSC)
    if not result['titleNumber'] and idx_ins >= 0:
        j = find_title_line(idx_ins + 1, idx_ins + 8)
        if j >= 0:
            result['titleNumber'] = re.sub(r'\D', '', norm[j])

    idx_zona = find_label(LABEL_ZONA, idx_ins + 1 if idx_ins >= 0 else 0)
    zona_val_idx = -1
    if not result['zona'] and idx_zona >= 0:
        j = find_digit_line(idx_zona + 1, idx_zona + 8)
        if j >= 0:
            result['zona'] = norm[j].strip()
            zona_val_idx = j

    idx_secao = find_label(LABEL_SECAO, idx_zona + 1 if idx_zona >= 0 else 0)
    if not result['secao'] and idx_secao >= 0:
        start = max(idx_secao + 1, zona_val_idx + 1)
        j = find_digit_line(start, idx_secao + 9)
        if j >= 0:
            result['secao'] = norm[j].strip()

    # --- data de nascimento: preferir a data perto do rótulo NASCIMENTO ---
    idx_nasc = find_label(r'NASC|RASC|MASO|Nsc')
    if idx_nasc >= 0:
        for j in range(idx_nasc, min(idx_nasc + 4, len(lines))):
            m = re.search(r'(\d{2})/(\d{2})/(\d{4})', norm[j])
            if m:
                result['dataNascimento'] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                break

    # --- passada 3: nome = linha(s) seguinte(s) ao rótulo (junta até 2 linhas) ---
    if not result['nome']:
        idx = find_label(LABEL_NOME)
        if idx >= 0:
            parts = []
            for j in range(idx + 1, min(idx + 4, len(lines))):
                if is_noise(norm[j]):
                    break
                parts.append(lines[j].strip())
                if nome_quality(' '.join(parts)) >= 20:
                    break
            if parts:
                result['nome'] = ' '.join(parts)

    # --- passada 4: melhor linha "parecida com nome" do texto inteiro ---
    # (não a primeira: avalia todas e fica com a de melhor qualidade)
    if not result['nome'] or nome_quality(result['nome']) < 20:
        best_name, best_q = '', -1
        for i, l in enumerate(norm):
            if is_noise(l):
                continue
            q = nome_quality(lines[i].strip())
            if q > best_q:
                best_name, best_q = lines[i].strip(), q
        if best_q > 0:
            result['nome'] = best_name

    result['nome'] = clean_nome(result['nome']) or result['nome']

    # --- passada 5: título como número "nu" (rótulo ilegível) —
    # bloco de 10-14 dígitos com agrupamento por espaços é quase sempre o nº do título
    if not result['titleNumber']:
        for l in lines:
            digits = re.sub(r'\D', '', l)
            if 10 <= len(digits) <= 14 and re.search(r'\d\s+\d', l):
                result['titleNumber'] = digits
                break

    return result


def crop_under_label(img, bbox):
    """Recorta a região abaixo/à direita do rótulo, ampliada 2x."""
    iw, ih = img.size
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = x2 - x1
    h = y2 - y1
    cx1 = max(0, int(x1 - w * 0.15))
    cx2 = min(iw, int(x2 + w * 2.5))
    cy1 = max(0, int(y1 - h * 0.25))
    cy2 = min(ih, int(y2 + h * 1.2))
    crop = img.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    return crop


def reocr_title_number(img, bbox):
    """Re-OCR do nº de inscrição: várias janelas de recorte × binarização,
    fica com a leitura de maior confiança."""
    iw, ih = img.size
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = x2 - x1
    h = y2 - y1
    windows = [
        (-0.5, 2.0, 0.0, 2.0),
        (0.0, 1.0, 0.0, 1.5),
        (0.0, 2.5, 0.0, 1.2),
    ]
    best = None
    for dx1, dx2, dy1, dy2 in windows:
        cx1 = max(0, int(x1 + w * dx1))
        cx2 = min(iw, int(x2 + w * dx2))
        cy1 = max(0, int(y1 + h * dy1))
        cy2 = min(ih, int(y2 + h * dy2))
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        gray = ImageOps.autocontrast(crop.convert('L'))
        for threshold in (140, 180):
            b = gray.point(lambda p, t=threshold: 255 if p > t else 0)
            for _, tx, cf in reader.readtext(np.array(b), detail=1, paragraph=False, allowlist='0123456789'):
                digits = re.sub(r'\D', '', tx)
                if 10 <= len(digits) <= 14 and (best is None or cf > best[1]):
                    best = (digits, cf)
    return best[0] if best else None


def reocr_small_number(img, bbox, exclude=''):
    """Re-OCR de zona/seção (1-4 dígitos) na região sob o rótulo."""
    crop = crop_under_label(img, bbox)
    gray = ImageOps.autocontrast(crop.convert('L'))
    best = None
    for threshold in (140, 180):
        b = gray.point(lambda p, t=threshold: 255 if p > t else 0)
        for _, tx, cf in reader.readtext(np.array(b), detail=1, paragraph=False, allowlist='0123456789'):
            digits = re.sub(r'\D', '', tx)
            if 1 <= len(digits) <= 4 and digits != exclude and (best is None or cf > best[1]):
                best = (digits, cf)
    return best[0] if best else None


def titulo_valido(t):
    """Valida o dígito verificador do nº de título de eleitor (algoritmo TSE).
    12 dígitos: 8 (sequencial) + 2 (UF 01-28) + 2 (DV). Retorno True/False."""
    t = _clean_num(t)
    if len(t) != 12:
        return False
    d = [int(c) for c in t]
    uf = d[8] * 10 + d[9]
    if not (1 <= uf <= 28):
        return False
    s = sum(a * b for a, b in zip(d[:8], range(2, 10)))
    dv1 = (s % 11) % 10
    s2 = d[8] * 7 + d[9] * 8 + dv1 * 9
    dv2 = (s2 % 11) % 10
    return d[10] == dv1 and d[11] == dv2


def nome_quality(s):
    """Pontua um candidato a nome: mais palavras capitalizadas puras = melhor.
    Rejeita lixo de marca d'água/assinatura (dígitos no meio, 1 palavra só)."""
    if not s or re.search(r'\d', s):
        return -1
    tokens = s.split()
    valid = sum(1 for t in tokens if re.fullmatch(r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ']+", t))
    if valid < 2:  # nome completo tem ao menos nome+sobrenome
        return -1
    return valid * 10 + len(s)


# Rótulos que a IA/OCR às vezes cola no começo do nome ("Nome completo: Joao")
NOME_LABEL_RE = re.compile(
    r'^\s*(?:nome\s*completo|nome\s*do\s*eleitor[ae]?|nome\s*do\s*cadastrado|nome)'
    r'\s*:\s*',
    re.IGNORECASE,
)
# "Nome" sem dois-pontos colado ao valor ("Nome SUELY SANTOS") — só remove
# se o resto for composto só de palavras/pontuação de nome (sem dígitos)
NOME_BARE_RE = re.compile(r"^\s*Nome\s+(?=[A-ZÀ-ÿ][A-Za-zÀ-ÿ'\s]+$)", re.IGNORECASE)


def clean_nome(s):
    """Remove rótulos ('Nome completo:', 'Nome:', etc.) e sujeira do nome."""
    s = str(s or '').strip()
    prev = None
    while prev != s:
        prev = s
        s = NOME_LABEL_RE.sub('', s)
        s = NOME_BARE_RE.sub('', s)
    return s.strip(' :\t')


def main_is_label(t):
    return bool(re.search(r'MUNIC|UF\b|DATA|EMISSAO|CORRESPOND|AUTENTIC|TRIBUNAL|'
                          r'ORIENTAC|APTOS|ELEITORES|ELEITORAS|BIOMETRIA|ULTIMA|OPERAC', t))


def most_common(values):
    from collections import Counter
    vals = [v for v in values if v]
    if not vals:
        return ''
    return Counter(vals).most_common(1)[0][0]


KEYWORDS = ['TITULO', 'ELEITORAL', 'REPUBLICA', 'FEDERAT', 'BRASIL', 'ZONA',
            'SECAO', 'INSCRICAO', 'ELEITOR', 'NASCIMENTO', 'MUNICIPIO', 'NOME']


def rank_angles(img, step=9, span=36):
    """Foto de celular vem com perspectiva — ângulos diferentes leem partes
    diferentes. Pontua por sinais úteis pra extração: palavras-chave do
    documento + datas + blocos de dígitos (título/zona/seção)."""
    small = img.resize((img.width // 4, img.height // 4), Image.LANCZOS)
    scored = []
    for angle in range(-span, span + 1, step):
        rotated = small.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        dets = reader.readtext(np.array(rotated), detail=1, paragraph=False)
        if not dets:
            continue
        texts_up = [strip_accents(tx).upper() for _, tx, _ in dets]
        kw = sum(1 for tx in texts_up for k in KEYWORDS if k in tx)
        dates = sum(1 for tx in texts_up if re.search(r'\d{2}/\d{2}/\d{4}', tx))
        title_runs = sum(1 for tx in texts_up if 10 <= len(re.sub(r'\D', '', tx)) <= 14)
        digit_lines = sum(1 for tx in texts_up if re.fullmatch(r'\d{1,4}', tx.strip()))
        mean_conf = float(np.mean([cf for _, _, cf in dets]))
        score = kw * 10 + dates * 5 + title_runs * 40 + digit_lines * 3 + mean_conf * 10
        scored.append((score, angle))
    scored.sort(reverse=True)
    # ângulo 0 sempre primeiro: foto reta resolve em 1 passada
    others = [a for _, a in scored if a != 0][:2]
    return [0] + others


_load_env()
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')


def _parse_model_env(raw):
    models = [m.strip() for m in str(raw or '').split(',') if m.strip()]
    return models


# Default pool uses the names provided by the user. Keep it env-overridable
# because OpenRouter model IDs can differ from storefront labels.
OPENROUTER_MODELS = _parse_model_env(os.environ.get(
    'OPENROUTER_MODELS',
    ','.join([
        'minimax/minimax-m3:free',
        'thinkingmachines/inkling-small:free',
        'dots-studio/dots-3-note-preview:free',
    ]),
))
MODEL_REQUEST_TIMEOUT_SECONDS = float(os.environ.get('OPENROUTER_TIMEOUT_SECONDS', '60'))
MODEL_RETRY_LIMIT = int(os.environ.get('OPENROUTER_RETRY_LIMIT', '3'))
MODEL_COOLDOWN_SECONDS = float(os.environ.get('OPENROUTER_COOLDOWN_SECONDS', '45'))


@dataclass
class ModelSlot:
    model: str
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ''


MODEL_SLOTS = [ModelSlot(model=m) for m in OPENROUTER_MODELS]


def send_discord_alert(event, details):
    if not DISCORD_WEBHOOK_URL:
        return
    body = _json.dumps({
        'content': (
            f"[ocr-service] {event}\n"
            f"```json\n{_json.dumps(details, ensure_ascii=False, indent=2)[:1600]}\n```"
        )
    }).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return
    except Exception as exc:
        print(f'[discord] erro ao enviar alerta: {exc!r}', flush=True)


def pick_available_slot(slots, now=None):
    now = time.time() if now is None else now
    for slot in slots:
        if slot.cooldown_until <= now:
            return slot
    return None


def mark_slot_failure(slot, reason, now=None, cooldown_seconds=MODEL_COOLDOWN_SECONDS):
    now = time.time() if now is None else now
    slot.cooldown_until = now + cooldown_seconds
    slot.consecutive_failures += 1
    slot.last_error = reason


def mark_slot_success(slot):
    slot.cooldown_until = 0.0
    slot.consecutive_failures = 0
    slot.last_error = ''


def handle_model_exception(slot, exc, now=None):
    code = getattr(exc, 'code', None)
    if code == 429:
        reason = 'http_429'
        cooldown = max(MODEL_COOLDOWN_SECONDS, 60.0)
        event = 'provider_rate_limit'
    elif isinstance(exc, TimeoutError):
        reason = 'timeout'
        cooldown = MODEL_COOLDOWN_SECONDS
        event = 'provider_timeout'
    else:
        reason = f'http_{code}' if code else 'provider_error'
        cooldown = MODEL_COOLDOWN_SECONDS
        event = 'provider_error'
    mark_slot_failure(slot, reason, now=now, cooldown_seconds=cooldown)
    send_discord_alert(event, {'model': slot.model, 'reason': reason, 'code': code})
    raise exc


def _default_review_draft():
    return {
        'type': 'titulo',
        'fields': {
            'nome': '',
            'dataNascimento': '',
            'gender': '',
            'age': '',
            'titleNumber': '',
            'zona': '',
            'secao': '',
            'municipio': '',
            'uf': '',
        },
    }


def _build_ai_prompt():
    return (
        "Analise a imagem e responda APENAS um JSON válido.\n"
        "Se for um Título de Eleitor individual, retorne:\n"
        '{"type":"titulo","fields":{"nome":"","dataNascimento":"DD/MM/AAAA","gender":"M ou F (infira pelo nome baseando-se no contexto brasileiro)","titleNumber":"somente dígitos","zona":"número","secao":"número","municipio":"","uf":"sigla"}}\n'
        "Se for uma lista/caderneta com vários eleitores, retorne:\n"
        '{"type":"lista","voters":[{"nome":"","telefone":"somente dígitos","titleNumber":"somente dígitos","secao":"número","zona":"número","bairro":"","gender":"M ou F (infira pelo nome baseando-se no contexto brasileiro)"}]}\n'
        "Não escreva texto extra. Não invente valores além de inferir o gênero pelo nome."
    )


def _extract_json_object(content):
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        content = '\n'.join(text_parts)
    m = re.search(r'\{.*\}', str(content or ''), re.DOTALL)
    if not m:
        raise ValueError('Resposta sem JSON')
    return _json.loads(m.group(0))


def _normalize_ai_payload(parsed):
    if parsed.get('type') == 'lista':
        voters = []
        for voter in parsed.get('voters', []):
            if not isinstance(voter, dict):
                continue
            
            gender = str(voter.get('gender') or voter.get('genero') or '').strip().upper()
            if gender not in ('M', 'F'):
                gender = ''
                
            voters.append({
                'nome': clean_nome(voter.get('nome')),
                'telefone': _clean_num(voter.get('telefone')),
                'titleNumber': _clean_num(voter.get('titleNumber')),
                'secao': _clean_num(voter.get('secao')),
                'zona': _clean_num(voter.get('zona')),
                'bairro': str(voter.get('bairro', '') or '').strip(),
                'gender': gender,
                'titleValid': titulo_valido(_clean_num(voter.get('titleNumber')))
                    if _clean_num(voter.get('titleNumber')) else None,
            })
        return {'type': 'lista', 'voters': voters}

    fields = _default_review_draft()['fields']
    src = parsed.get('fields') if isinstance(parsed.get('fields'), dict) else parsed
    for key in fields:
        value = src.get(key, '') if isinstance(src, dict) else ''
        if key == 'nome':
            fields[key] = clean_nome(value)
        elif key in {'titleNumber', 'zona', 'secao'}:
            fields[key] = _clean_num(value)
        elif key == 'gender':
            v = str(value or src.get('genero') or '').strip().upper()
            fields[key] = v if v in ('M', 'F') else ''
        elif key == 'age':
            continue
        else:
            fields[key] = str(value or '').strip()
            
    # Calcula a idade automaticamente
    if fields.get('dataNascimento'):
        try:
            from datetime import datetime
            birth = datetime.strptime(fields['dataNascimento'], '%d/%m/%Y')
            today = datetime.today()
            fields['age'] = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except Exception:
            fields['age'] = ''
            
    return {'type': 'titulo', 'fields': fields}


def _encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def run_model_request(slot, img_b64, prompt):
    body = _json.dumps({
        'model': slot.model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
            ],
        }],
        'max_tokens': 3000,
    }).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        },
    )
    with urllib.request.urlopen(req, timeout=MODEL_REQUEST_TIMEOUT_SECONDS) as resp:
        return _json.loads(resp.read())


def build_review_payload_from_response(data, slot, latency_ms):
    content = data['choices'][0]['message']['content']
    parsed = _extract_json_object(content)
    normalized = _normalize_ai_payload(parsed)
    return {
        'rawResult': parsed,
        'reviewDraft': normalized,
        'modelUsed': slot.model,
        'latencyMs': latency_ms,
        'slotState': asdict(slot),
    }


def process_image_ai_only(img):
    if not OPENROUTER_API_KEY:
        raise RuntimeError('OPENROUTER_API_KEY ausente')
    img_b64 = _encode_image(img)
    prompt = _build_ai_prompt()
    last_exc = None
    for attempt in range(1, MODEL_RETRY_LIMIT + 1):
        slot = pick_available_slot(MODEL_SLOTS)
        if slot is None:
            send_discord_alert('model_pool_exhausted', {'attempt': attempt, 'models': OPENROUTER_MODELS})
            time.sleep(1)
            continue
        started = time.time()
        try:
            data = run_model_request(slot, img_b64, prompt)
            payload = build_review_payload_from_response(data, slot, int((time.time() - started) * 1000))
            payload['attemptCount'] = attempt
            mark_slot_success(slot)
            return payload
        except urllib.error.HTTPError as exc:
            last_exc = exc
            handle_model_exception(slot, exc)
        except Exception as exc:
            last_exc = exc
            mark_slot_failure(slot, type(exc).__name__)
            send_discord_alert(
                'provider_exception',
                {'model': slot.model, 'attempt': attempt, 'error': repr(exc)[:400]},
            )
    raise RuntimeError(f'falha ao processar imagem no pool de IA: {last_exc!r}')

# banco da fila de jobs (mesmo Postgres da aplicação)
def _sanitize_dsn(dsn):
    """Remove query params (ex.: ?schema=public do Prisma) e re-encoda a
    senha — libpq (psycopg2) não tolera caracteres especiais como '@'."""
    dsn = (dsn or '').split('?')[0]
    try:
        from urllib.parse import urlsplit, urlunsplit, quote
        p = urlsplit(dsn)
        host = p.hostname or ''
        if p.port:
            host += f':{p.port}'
        cred = ''
        if p.username:
            cred = quote(p.username, safe='')
            if p.password:
                cred += ':' + quote(p.password, safe='')
            cred += '@'
        return urlunsplit((p.scheme, cred + host, p.path, '', ''))
    except Exception:
        return dsn

DB_DSN = _sanitize_dsn(os.environ.get('DATABASE_URL') or 'postgresql://pedroroger@localhost:5432/sistema_eleitoral')
JOBS_DIR = os.path.join(os.path.dirname(__file__), 'jobs_images')
os.makedirs(JOBS_DIR, exist_ok=True)


def db():
    return psycopg2.connect(DB_DSN)


def _decode_job_result(value):
    if value and isinstance(value, str):
        try:
            return _json.loads(value)
        except Exception:
            return {'raw': value}
    return value or {}


def fetch_job_row(job_id):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('SELECT * FROM ocr_jobs WHERE id = %s', (job_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_job_result_payload(job_id, payload, status=None, error=None):
    conn = db()
    cur = conn.cursor()
    if status is None:
        cur.execute(
            'UPDATE ocr_jobs SET result = %s, error = %s WHERE id = %s',
            (_json.dumps(payload), error, job_id),
        )
    else:
        cur.execute(
            'UPDATE ocr_jobs SET status = %s, result = %s, error = %s, "finishedAt" = now() WHERE id = %s',
            (status, _json.dumps(payload), error, job_id),
        )
    conn.commit()
    cur.close()
    conn.close()


def mark_job_confirmed(job_id, payload):
    update_job_result_payload(job_id, payload, status='confirmed')


async def ocr_worker():
    """Consome a fila ocr_jobs: 1 job por vez e envia para o pool de IA."""
    while True:
        try:
            conn = db()
            conn.autocommit = True
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "UPDATE ocr_jobs SET status = 'processing' "
                'WHERE id = (SELECT id FROM ocr_jobs WHERE status = %s ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED) '
                'RETURNING id, "imagePath", filename',
                ('queued',),
            )
            job = cur.fetchone()
            cur.close()
            conn.close()
            if not job:
                await asyncio.sleep(1)
                continue

            job_id = job['id']
            print(f'[worker] job {job_id} iniciado ({job["filename"]})', flush=True)
            started = time.time()
            try:
                img = Image.open(job['imagePath'])
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                result = await asyncio.to_thread(process_image_ai_only, img)
                update_job_result_payload(job_id, result, status='review_required')
                print(f'[worker] job {job_id} concluído em {time.time() - started:.0f}s', flush=True)
            except Exception as exc:
                print(f'[worker] job {job_id} ERRO: {exc!r}', flush=True)
                update_job_result_payload(job_id, {}, status='error', error=repr(exc)[:500])
        except Exception as exc:
            print(f'[worker] falha no loop: {exc!r}', flush=True)
            await asyncio.sleep(2)
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODEL = 'google/gemini-3.7-flash'  # vision barato e rápido


def ai_extract_fields(img, ocr_fields):
    """Fallback por IA: envia a imagem pra um modelo de visão (OpenRouter)
    pedindo só os campos que o OCR não conseguiu ler. Retorna o que achou."""
    if not OPENROUTER_API_KEY:
        return {}

    campos_faltando = [k for k in ('nome', 'dataNascimento', 'gender', 'titleNumber', 'zona', 'secao', 'municipio', 'uf')
                       if not ocr_fields.get(k)]
    if not campos_faltando:
        return {}

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "Esta é uma foto de um Título de Eleitor brasileiro. "
        f"O OCR não conseguiu ler estes campos: {', '.join(campos_faltando)}.\n"
        "Leia a imagem com atenção (inclua áreas com selo/marca d'água ou levemente tortas) "
        "e extraia APENAS os campos faltantes.\n"
        "Responda APENAS um JSON válido, sem texto extra, no formato:\n"
        '{"nome": "", "dataNascimento": "DD/MM/AAAA", "gender": "M ou F (infira pelo nome)", "titleNumber": "somente dígitos", '
        '"zona": "número", "secao": "número", "municipio": "", "uf": "sigla com 2 letras"}\n'
        "Se um campo não estiver visível ou não tiver certeza, devolva string vazia para ele.\n"
        "Não invente valores além de inferir o gênero."
    )

    body = _json.dumps({
        'model': OPENROUTER_MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ],
        }],
        'max_tokens': 2000,
    }).encode()

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = _json.loads(resp.read())
        content = data['choices'][0]['message']['content']
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            print('[IA] resposta sem JSON:', content[:200])
            return {}
        parsed = _json.loads(m.group(0))
        # devolve apenas os campos que faltavam e a IA preencheu
        out = {}
        for k in campos_faltando:
            v = str(parsed.get(k, '') or '').strip()
            if k == 'nome':
                v = clean_nome(v)
            if v:
                out[k] = v
        print('[IA] preencheu:', out)
        return out
    except Exception as exc:
        print('[IA] erro:', repr(exc))
        return {}


def looks_like_voter_list(text):
    """Detecta caderneta de campo: várias fichas manuscritas com
    Nome:/Endereço:/Telefone:/Nº Título: repetidos (com variações de OCR)."""
    t = strip_accents(text).upper()
    nomes = len(re.findall(r'NOM[EC]\s*:', t))
    tels = len(re.findall(r'TE[L1][CCLL]?F?[O0]?NE|TC[L1][O0]NE|T[E1]LEF[O0]NE', t))
    titulos = len(re.findall(r'T[L1]TU[LO]{1,2}\s*:|N[O0]\s*T[L1]TU[LO]{1,2}', t))
    return nomes >= 2 and (tels >= 1 or titulos >= 2)


def _clean_num(s):
    return re.sub(r'\D', '', str(s or ''))


def ai_extract_list(img):
    """Fallback por IA para cadernetas: extrai TODAS as fichas de eleitores
    (nome, telefone, título, seção, zona, bairro). Retorna lista normalizada."""
    if not OPENROUTER_API_KEY:
        return []

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
        "IMPORTANTE: cada campo pertence à ficha em que aparece — nunca misture telefone, "
        "título, seção ou zona de uma pessoa com outra. Se uma ficha estiver cortada/incompleta "
        "na imagem, inclua apenas se tiver ao menos nome E título (senão pule).\n"
        "No campo 'nome' devolva SOMENTE o nome da pessoa — nunca inclua o rótulo "
        "('Nome:', 'Nome completo:', 'Endereço' etc.).\n"
        "Telefone: 10-11 dígitos com DDD. Título: 10-12 dígitos, sem pontos. "
    )

    body = _json.dumps({
        'model': OPENROUTER_MODEL,
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
        OPENROUTER_URL,
        data=body,
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
        content = data['choices'][0]['message']['content']
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            print('[IA-lista] resposta sem JSON:', content[:200])
            return []
        parsed = _json.loads(m.group(0))
        voters, seen = [], set()
        for v in parsed.get('voters', []):
            if not isinstance(v, dict):
                continue
            item = {
                'nome': clean_nome(v.get('nome')),
                'telefone': _clean_num(v.get('telefone')),
                'titleNumber': _clean_num(v.get('titleNumber')),
                'secao': _clean_num(v.get('secao')),
                'zona': _clean_num(v.get('zona')),
                'bairro': str(v.get('bairro', '') or '').strip(),
                # DV TSE: False = dígitos não batem (leitura de mão ambígua) → UI avisa
                'titleValid': titulo_valido(_clean_num(v.get('titleNumber')))
                    if _clean_num(v.get('titleNumber')) else None,
            }
            filled = sum(1 for k in ('nome', 'telefone', 'titleNumber', 'secao', 'zona') if item[k])
            if filled < 2:
                continue
            key = item['titleNumber'] or f"{item['nome']}|{item['telefone']}"
            if key and key in seen:
                continue
            seen.add(key)
            voters.append(item)
        print(f'[IA-lista] extraídas {len(voters)} fichas')
        return voters
    except Exception as exc:
        print('[IA-lista] erro:', repr(exc))
        return []


def quick_list_check(img):
    """Detecção barata de caderneta: OCR em resolução reduzida (max 900px,
    ~3x mais rápido que a passada completa) só pra achar os rótulos
    Nome:/Telefone:/Título repetidos. Evita pagar OCR full-res só pra
    descobrir que a foto é uma lista."""
    try:
        w, h = img.size
        scale = 900 / max(w, h)
        small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS) if scale < 1 else img
        dets = reader.readtext(np.array(small), detail=1, paragraph=False)
        text = '\n'.join(t for _, t in dets)
        return looks_like_voter_list(text)
    except Exception:
        return False


def process_image_full(img):
    # normaliza pra 1200px no maior lado (upscale também):
    # imagens pequenas têm texto pequeno demais pro OCR.
    # 1200 em vez de 1600: ~35% mais rápido com a MESMA extração
    # (validado em título físico e e-Título — zona/seção/nome/título idênticos)
    max_side = 1200
    w, h = img.size
    scale = max_side / max(w, h)
    if abs(scale - 1.0) > 0.05:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # === atalho: caderneta detectada na passada rápida → IA direto ===
    # (pula o OCR full-res + ângulos + re-OCR regional, que é o que demora)
    if OPENROUTER_API_KEY:
        qw, qh = img.size
        qs = 900 / max(qw, qh)
        quick_img = img.resize((int(qw * qs), int(qh * qs)), Image.LANCZOS) if qs < 1 else img
        quick_txt = '\n'.join(
            t for _, t, _ in reader.readtext(np.array(quick_img), detail=1, paragraph=False)
        )
        if looks_like_voter_list(quick_txt):
            voters = ai_extract_list(img)
            if voters:
                return {'type': 'lista', 'voters': voters, 'rawText': quick_txt, 'aiUsed': True}

    def run_ocr(angle):
        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        detections = reader.readtext(np.array(rotated), detail=1, paragraph=False)
        texts = [(d[0], d[1]) for d in detections]
        full_text = '\n'.join(t for _, t in texts)
        return rotated, texts, full_text

    def fix_title_regional(rotated, texts):
        for bbox, text in texts:
            t = strip_accents(text).upper()
            if re.search(LABEL_INSC, t) and not re.search(r'\bZO\w{0,2}\b', t) and len(t) < 14:
                found = reocr_title_number(rotated, bbox)
                return found
        return None

    def fill_from(fields):
        for k in ('nome', 'dataNascimento', 'titleNumber', 'zona', 'secao', 'municipio', 'uf'):
            v = fields.get(k)
            if v and not merged.get(k):
                merged[k] = v

    def complete():
        return not [k for k in ('nome', 'dataNascimento', 'gender', 'titleNumber', 'zona', 'secao', 'municipio', 'uf')
                if not merged.get(k)]

    def faltando():
        return [k for k in ('nome', 'dataNascimento', 'gender', 'titleNumber', 'zona', 'secao', 'municipio', 'uf')
                if not merged.get(k)]

    merged = {'nome': '', 'dataNascimento': '', 'gender': '', 'age': '', 'titleNumber': '', 'zona': '', 'secao': '', 'municipio': '', 'uf': ''}
    data_candidates = []
    zona_candidates = []
    secao_candidates = []
    best_text = ''
    ai_used = False

    # === passo 1: OCR no ângulo 0 (foto como tirada) ===
    rotated0, texts0, full0 = run_ocr(0)
    best_text = full0
    fields = extract_fields(full0)
    if nome_quality(fields['nome']) > nome_quality(merged['nome']):
        merged['nome'] = fields['nome']
    for k in ('dataNascimento', 'titleNumber', 'zona', 'secao', 'uf'):
        if fields.get(k) and not merged.get(k):
            merged[k] = fields[k]
    if len(fields['municipio'].split()) * 10 + len(fields['municipio']) > len(merged['municipio'].split()) * 10 + len(merged['municipio']) and fields['municipio']:
        merged['municipio'] = fields['municipio']
    if fields['dataNascimento']:
        data_candidates.append(fields['dataNascimento'])
    if fields['zona']:
        zona_candidates.append(fields['zona'])
    if fields['secao']:
        secao_candidates.append(fields['secao'])

    # re-OCR regional binarizado do nº de inscrição (corrige dígitos errados).
    # Só substitui se a leitura atual for INVÁLIDA (DV TSE) e a nova for válida —
    # evita que uma leitura de alta confiança porém errada sobrescreva a boa.
    if merged['titleNumber'] and not titulo_valido(merged['titleNumber']):
        found = fix_title_regional(rotated0, texts0)
        if found and titulo_valido(found):
            merged['titleNumber'] = found

    # leitura de título que não passa no DV: descarta e deixa a IA re-ler
    if merged['titleNumber'] and not titulo_valido(merged['titleNumber']):
        print(f"[worker] titulo {merged['titleNumber']} falhou no DV — limpando pra IA")
        merged['titleNumber'] = ''

    # === caderneta de campo (vários eleitores por foto): extrai lista via IA ===
    if looks_like_voter_list(full0) and OPENROUTER_API_KEY:
        voters = ai_extract_list(img)
        if voters:
            return {'type': 'lista', 'voters': voters, 'rawText': best_text, 'aiUsed': True}

    if complete():
        return {'fields': merged, 'rawText': best_text, 'aiUsed': False}

    # === passo 2: IA para o que faltar (rápida, resolve fotos difíceis) ===
    if OPENROUTER_API_KEY and faltando:
        ai_fields = ai_extract_fields(img, merged)
        for k, v in ai_fields.items():
            if v and not merged.get(k):
                merged[k] = v
        if ai_fields:
            ai_used = True
        faltando = [k for k in ('nome', 'dataNascimento', 'gender', 'titleNumber', 'zona', 'secao', 'municipio', 'uf') if not merged.get(k)]
        if len(faltando) <= 1:
            return {'fields': merged, 'rawText': best_text, 'aiUsed': ai_used}

    # === passo 3: ângulos extras (foto torta) + regional em cada um ===
    for angle in rank_angles(img)[:3]:
        if angle == 0:
            continue
        rotated, texts, full_text = run_ocr(angle)
        if len(full_text) > len(best_text):
            best_text = full_text
        fields = extract_fields(full_text)
        if nome_quality(fields['nome']) > nome_quality(merged['nome']):
            merged['nome'] = fields['nome']
        for k in ('dataNascimento', 'titleNumber', 'zona', 'secao', 'uf'):
            if fields.get(k) and not merged.get(k):
                merged[k] = fields[k]
        if len(fields['municipio'].split()) * 10 + len(fields['municipio']) > len(merged['municipio'].split()) * 10 + len(merged['municipio']) and fields['municipio']:
            merged['municipio'] = fields['municipio']
        if fields['dataNascimento']:
            data_candidates.append(fields['dataNascimento'])
        if fields['zona']:
            zona_candidates.append(fields['zona'])
        if fields['secao']:
            secao_candidates.append(fields['secao'])

        if not merged['titleNumber']:
            found = fix_title_regional(rotated, texts)
            if found and titulo_valido(found):
                merged['titleNumber'] = found
        if complete():
            break

    # nascimento: ano mais antigo entre as datas (data de emissão é recente)
    if data_candidates:
        def ano(d):
            try:
                return int(d.split('/')[2])
            except Exception:
                return 9999
        melhor = min(data_candidates, key=ano)
        if ano(melhor) <= 2009 or len(data_candidates) == 1:
            merged['dataNascimento'] = melhor
            try:
                from datetime import datetime
                b = datetime.strptime(melhor, '%d/%m/%Y')
                t = datetime.today()
                merged['age'] = t.year - b.year - ((t.month, t.day) < (b.month, b.day))
            except Exception:
                merged['age'] = ''

    # fallback zona: linha nua após o título no melhor texto
    if not merged['zona'] and merged['titleNumber']:
        tl = best_text.split('\n')
        for i, l in enumerate(tl):
            digits = re.sub(r'\D', '', l)
            if 10 <= len(digits) <= 14 and digits == merged['titleNumber']:
                for j in range(i + 1, min(i + 5, len(tl))):
                    if re.fullmatch(r'\d{1,4}', strip_accents(tl[j]).strip()):
                        merged['zona'] = strip_accents(tl[j]).strip()
                        break
                break

    # === passo 4: IA final para o que ainda faltar ===
    faltando = [k for k in ('nome', 'dataNascimento', 'gender', 'titleNumber', 'zona', 'secao', 'municipio', 'uf') if not merged.get(k)]
    if faltando and OPENROUTER_API_KEY:
        ai_fields = ai_extract_fields(img, merged)
        for k, v in ai_fields.items():
            if v and not merged.get(k):
                merged[k] = v
        if ai_fields:
            ai_used = True

    return {'fields': merged, 'rawText': best_text, 'aiUsed': ai_used}


from fastapi.responses import JSONResponse

JOB_QUEUE = asyncio.Queue(maxsize=200)


@app.post("/jobs")
async def create_job(file: UploadFile = File(...), createdby: int = 0, createdbyname: str = ''):
    """Enfileira uma foto pra processamento em background. Retorna jobId na hora."""
    contents = await file.read()
    job_id = None
    path = os.path.join(JOBS_DIR, f'{_uuid.uuid4().hex}.jpg')
    with open(path, 'wb') as f:
        f.write(contents)

    conn = db()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO ocr_jobs (status, "createdBy", "createdByName", filename, "imagePath") '
        'VALUES (%s, %s, %s, %s, %s) RETURNING id',
        ('queued', createdby or None, createdbyname or None, file.filename, path),
    )
    job_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return {'jobId': job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: int):
    row = fetch_job_row(job_id)
    if not row:
        return JSONResponse({'error': 'Job não encontrado.'}, status_code=404)
    row['result'] = _decode_job_result(row.get('result'))
    return {'job': row}


@app.patch("/jobs/{job_id}/review")
async def save_review(job_id: int, body: dict):
    row = fetch_job_row(job_id)
    if not row:
        return JSONResponse({'error': 'Job não encontrado.'}, status_code=404)
    result_payload = _decode_job_result(row.get('result'))
    result_payload['reviewDraft'] = body.get('reviewDraft') or result_payload.get('reviewDraft') or _default_review_draft()
    result_payload['reviewUpdatedAt'] = int(time.time())
    update_job_result_payload(job_id, result_payload, status='review_required')
    return {'jobId': job_id, 'status': 'review_required'}


@app.post("/jobs/{job_id}/confirm")
async def confirm_job(job_id: int):
    row = fetch_job_row(job_id)
    if not row:
        return JSONResponse({'error': 'Job não encontrado.'}, status_code=404)
    result_payload = _decode_job_result(row.get('result'))
    review_draft = result_payload.get('reviewDraft')
    if not review_draft:
        return JSONResponse({'error': 'Job sem rascunho revisado.'}, status_code=400)
    result_payload['confirmedAt'] = int(time.time())
    mark_job_confirmed(job_id, result_payload)
    return {'jobId': job_id, 'status': 'confirmed'}


@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """Log/auditoria dos últimos jobs."""
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('SELECT id, status, "createdByName", filename, error, "createdAt", "finishedAt" '
                'FROM ocr_jobs ORDER BY id DESC LIMIT %s', (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {'jobs': rows}


@app.on_event('startup')
async def startup():
    asyncio.create_task(ocr_worker())
    print('[worker] fila iniciada', flush=True)
