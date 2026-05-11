import os, io, time, csv, random, sys, types

# pkg_resources shim — required by tensorflow_hub on Python 3.12 in some envs
try:
    import pkg_resources
except ImportError:
    _mod = types.ModuleType('pkg_resources')
    _mod.parse_version = lambda v: tuple(int(x) for x in str(v).split('.')[:3] if x.isdigit())
    sys.modules['pkg_resources'] = _mod

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF boot noise
import numpy as np
import librosa
from scipy.stats import entropy
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)

# GLB files need this MIME type or model-viewer won't load them
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
import mimetypes
mimetypes.add_type('model/gltf-binary', '.glb')

# Load YAMNet once at startup

print('Loading YAMNet model...')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names from the YAMNet repo (static CSV)
import requests
import tempfile
CLASS_MAP_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(requests.get(CLASS_MAP_URL).content)
    tmp.flush()
    class_names = [row['display_name'] for row in csv.DictReader(open(tmp.name))]

# Each category: list of keywords matched against class names (case-insensitive substring)
# Optional 'exclude' list prevents false matches (e.g. 'wind instrument' for wind)
CATEGORIES = {
    # NATURAL
    'rain':          {'keywords': ['rain', 'raindrop', 'drizzle']},
    'water':         {'keywords': ['stream', 'waterfall', 'ocean', 'wave', 'splash', 'drip', 'pour', 'trickle', 'liquid', 'water'], 'exclude': ['watering']},
    'thunder':       {'keywords': ['thunder', 'thunderstorm', 'lightning']},
    'fire':          {'keywords': ['fire', 'crackle', 'flame', 'burning']},
    'wind':          {'keywords': ['wind', 'breeze', 'rustling'], 'exclude': ['wind instrument', 'woodwind']},
    'amphibians':    {'keywords': ['frog', 'toad', 'croak']},
    'insects':       {'keywords': ['insect', 'cricket', 'mosquito', 'bee', 'wasp', 'cicada', 'fly, house']},
    'birds':         {'keywords': ['bird', 'chirp', 'tweet', 'squawk', 'pigeon', 'crow', 'owl', 'caw', 'coo', 'hoot', 'duck', 'goose', 'flapping wings']},
    'mammals':       {'keywords': ['dog', 'cat', 'horse', 'cattle', 'pig', 'sheep', 'wolf', 'lion', 'bark', 'meow', 'growl', 'roar', 'howl', 'neigh', 'moo', 'purr']},
    # HUMAN MADE
    'vehicles':      {'keywords': ['vehicle', 'car', 'truck', 'bus', 'motorcycle', 'train', 'aircraft', 'boat', 'traffic', 'motor vehicle', 'bicycle']},
    'voices':        {'keywords': ['speech', 'conversation', 'narration', 'shout', 'whispering', 'laughter', 'child speech', 'babbling', 'singing', 'yell', 'scream']},
    'footsteps':     {'keywords': ['footstep', 'walk, footstep', 'run', 'stomp']},
    'machinery':     {'keywords': ['machine', 'power tool', 'drill', 'saw', 'mechanical fan', 'industrial', 'jackhammer', 'compressor']},
    'music':         {'keywords': ['music', 'musical instrument', 'guitar', 'piano', 'drum', 'beat', 'rhythm'], 'exclude': ['bird', 'speech']},
    'object_moving': {'keywords': ['sliding door', 'clatter', 'thud', 'bang', 'knock', 'tap', 'click', 'rattle', 'rolling']},
    'bells':         {'keywords': ['bell', 'chime', 'church bell', 'cowbell', 'jingle', 'ding', 'gong', 'singing bowl'], 'exclude': ['doorbell']},
    'alarm_siren':   {'keywords': ['alarm', 'siren', 'buzzer', 'beep', 'civil defense']},
    'construction':  {'keywords': ['jackhammer', 'drilling', 'hammer', 'sawing', 'construction', 'demolition']},
    # BACKGROUND
    'low_freq':      {'keywords': ['hum', 'drone', 'rumble', 'low frequency']},
    'white_noise':   {'keywords': ['white noise', 'static', 'pink noise']},
    'wind_howl':     {'keywords': ['howl', 'whoosh', 'swoosh', 'gust'], 'exclude': ['dog', 'wolf']},
    'handling':      {'keywords': ['crinkle', 'rustle', 'microphone', 'handling noise']},
    'silence':       {'keywords': ['silence', 'quiet']},
}

def build_indices(entry):
    kws = [k.lower() for k in entry['keywords']]
    exs = [e.lower() for e in entry.get('exclude', [])]
    indices = []
    for i, name in enumerate(class_names):
        n = name.lower()
        if any(kw in n for kw in kws) and not any(ex in n for ex in exs):
            indices.append(i)
    return indices

CATEGORY_INDICES = {cat: build_indices(entry) for cat, entry in CATEGORIES.items()}

# Log what was matched so you can see it once at startup
for cat, idxs in CATEGORY_INDICES.items():
    matched = [class_names[i] for i in idxs]
    print(f'  {cat}: {matched}')

print('YAMNet ready.')

OUTPUT_DIR = 'OUTPUTS'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Prompt bank ───────────────────────────────────────────────────────────────
SHEET_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQOL-cy_b3u_us5w3C3023dDH5KDd3UpBSnDqtRijijz_rrFdczQlXZyuIZCfs4Ogq1RjVpMe5edrdU/pub?output=csv'
_bank_cache = None
_bank_time  = 0
BANK_TTL    = 300  # re-fetch after 5 minutes

def fetch_prompt_bank():
    global _bank_cache, _bank_time
    now = time.time()
    if _bank_cache is not None and now - _bank_time < BANK_TTL:
        return _bank_cache
    try:
        r = requests.get(SHEET_URL, timeout=5)
        r.raise_for_status()
        reader = csv.reader(io.StringIO(r.content.decode('utf-8')))
        rows   = list(reader)[1:]   # skip header row
        bank     = {}
        sections = {}   # keyed by column-1 section heading — separate from context keys
        special  = {}
        current_section = ''
        for row in rows:
            if len(row) < 2:
                continue
            section     = row[0].strip() if row[0].strip() else current_section
            current_section = section
            prompt      = row[1].strip()
            context_raw = row[2].strip() if len(row) > 2 else ''
            if not prompt:
                continue
            # Also index by section heading (column 1) so client can look up by e.g. 'SOUND'
            if current_section:
                sec_key = current_section.strip().upper()
                sections.setdefault(sec_key, [])
                if prompt not in sections[sec_key]:
                    sections[sec_key].append(prompt)
            ctx_upper = context_raw.upper()
            # Special fixed-position prompts
            if 'FIRST STATEMENT ALWAYS' in ctx_upper:
                h = time.localtime().tm_hour
                special['greeting'] = 'Good morning' if h < 12 else ('Good afternoon' if h < 17 else 'Good evening')
                continue
            if 'SECOND STATEMENT ALWAYS' in ctx_upper:
                special['second'] = prompt
                continue
            if 'ALWAYS THE LAST STATEMENT' in ctx_upper:
                special['last'] = prompt
                continue
            # Multi-context entries
            for ctx in [c.strip().upper() for c in context_raw.split('+') if c.strip()]:
                bank.setdefault(ctx, [])
                if prompt not in bank[ctx]:
                    bank[ctx].append(prompt)
        bank['special']  = special
        bank['_sections'] = sections   # column-1 section headings, non-destructive addition
        _bank_cache = bank
        _bank_time  = now
        sizes = ', '.join(f'{k}:{len(v)}' for k, v in bank.items() if k != 'special')
        print(f'[prompts] loaded — {sizes}')
        return bank
    except Exception as e:
        print(f'[prompts] fetch failed: {e}')
        return None

@app.route('/prompts')
def get_prompts():
    bank = fetch_prompt_bank()
    if bank is None:
        return jsonify({'error': 'unavailable'}), 503
    return jsonify(bank)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/sketch.js')
def serve_sketch():
    return send_from_directory('.', 'sketch.js')

@app.route('/<path:filename>')
def serve_static(filename):
    mime = 'model/gltf-binary' if filename.endswith('.glb') else None
    if mime:
        return send_from_directory('.', filename, mimetype=mime)
    return send_from_directory('.', filename)

def remap(value, in_min, in_max, out_min, out_max):
    t = (value - in_min) / (in_max - in_min + 1e-9)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)

def analyse_audio(y, sr):
    t0 = time.time()
    print(f'  audio length: {len(y)/sr:.1f}s, sr={sr}')
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    freq_mask = freqs >= 100
    spec_entropy = []
    for i in range(S.shape[1]):
        spec = S[freq_mask, i]
        total = np.sum(spec)
        if total > 0:
            prob = spec / total
            spec_entropy.append(entropy(prob, base=2))
    mean_entropy = float(np.mean(spec_entropy)) if spec_entropy else 8.0
    std_entropy  = float(np.std(spec_entropy))  if spec_entropy else 0.0
    print(f'  entropy done: {time.time()-t0:.2f}s')

    rms_frames = librosa.feature.rms(y=y)[0]
    mean_rms   = float(np.mean(rms_frames))
    print(f'  rms done: {time.time()-t0:.2f}s')

    y_short = y[:sr * 3] if len(y) > sr * 3 else y
    try:
        f0 = librosa.yin(y_short, fmin=60, fmax=2000,
                         frame_length=1024, hop_length=512)
        f0_valid = f0[(f0 > 60) & (f0 < 2000)]
        mean_f0 = float(np.median(f0_valid)) if len(f0_valid) > 0 else 200.0
    except Exception as e:
        print(f'  yin failed: {e}')
        mean_f0 = 200.0
    print(f'  pitch done: {time.time()-t0:.2f}s')

    flatness      = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = float(np.mean(flatness))
    print(f'  flatness done: {time.time()-t0:.2f}s')

    # YAMNet — per-category soundmark detection
    waveform = tf.constant(y, dtype=tf.float32)
    outputs = yamnet_model(waveform)
    # hub.load() returns a tuple or struct; handle both forms
    scores = outputs[0] if isinstance(outputs, (list, tuple)) else outputs['output_0']
    mean_scores = np.mean(np.array(scores), axis=0)

    THRESHOLD = 0.05
    soundmarks = {}
    detected_categories = []
    for cat, indices in CATEGORY_INDICES.items():
        if indices:
            score = float(np.max(mean_scores[indices]))
        else:
            score = 0.0
        soundmarks[cat] = round(score, 4)
        if score >= THRESHOLD:
            detected_categories.append(cat)

    print(f'  yamnet detected={detected_categories}, done: {time.time()-t0:.2f}s')

    params = {
        'numOrigins': int(remap(mean_entropy,  8.0, 10.5,  4,  16)),
        'lineWeight': round(remap(mean_rms,    0.0,  0.3,  0.5, 2.8), 3),
        'bass':       round(remap(std_entropy, 0.0,  1.2,  0.0, 1.0), 4),
        'treble':     round(remap(mean_f0,    80.0, 800.0, 1.0, 0.0), 4),
        'colorTemp':  round(remap(mean_flatness, 0.0, 0.3, 0.0, 1.0), 4),
    }
    measures = {
        'mean_entropy': round(mean_entropy,  4),
        'std_entropy':  round(std_entropy,   4),
        'mean_rms':     round(mean_rms,      4),
        'keynote_hz':   round(mean_f0,       2),
        'signal':       round(mean_flatness, 4),
    }
    print(f'  TOTAL: {time.time()-t0:.2f}s')
    return {'params': params, 'measures': measures, 'soundmarks': soundmarks, 'detected_categories': detected_categories}

@app.route('/analyse', methods=['POST'])
def analyse():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    audio_bytes = f.read()
    print(f'Received file: {len(audio_bytes)/1024:.0f} KB')
    print(f'First 20 bytes: {audio_bytes[:20]}')
    try:
        t0 = time.time()
        y_full, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        total_dur = len(y_full) / sr
        SEG = 30
        if total_dur >= SEG:
            start = random.uniform(0, total_dur - SEG)
            start_s = int(start * sr)
            y = y_full[start_s : start_s + int(SEG * sr)]
            segment_start = round(start, 3)
            print(f'Load done: {time.time()-t0:.2f}s, total={total_dur:.1f}s, segment={start:.1f}s\u2013{start+SEG:.1f}s')
        else:
            y = y_full[:int(10 * sr)] if len(y_full) > int(10 * sr) else y_full
            segment_start = 0.0
            print(f'Load done: {time.time()-t0:.2f}s, total={total_dur:.1f}s (short clip, using first {len(y)/sr:.1f}s)')
    except Exception as e:
        print(f'Librosa load error: {e}')
        return jsonify({'error': f'Could not decode audio: {str(e)}'}), 422
    result = analyse_audio(y, sr)
    result['segment_start'] = segment_start
    result['segment_duration'] = SEG if total_dur >= SEG else round(len(y) / sr, 3)
    print(f'Analysis result: {result}')
    return jsonify(result)

if __name__ == '__main__':
    print('-' * 50)
    print('  MILIEU server')
    print('  http://localhost:5000')
    print('-' * 50)
    app.run(debug=False, port=5000, threaded=False)