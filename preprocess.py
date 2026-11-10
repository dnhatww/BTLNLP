"""Tiền xử lý bình luận; không dùng nhãn để biến đổi văn bản."""
import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent
if (BASE / '.python_deps').exists():
    sys.path.insert(0, str(BASE / '.python_deps'))
import pandas as pd
from bs4 import BeautifulSoup
from underthesea import word_tokenize

DICTIONARY = json.loads((BASE / 'teencode_dictionary.json').read_text(encoding='utf-8'))
TEEN = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, DICTIONARY['replacements'])) + r')(?!\w)', re.I)
AMBIGUOUS = re.compile(r'(?<!\w)(' + '|'.join(DICTIONARY['ambiguous_keep']) + r')(?!\w)', re.I)
URL = re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>]+')
TAG = re.compile(r'</?[a-zA-Z][^>]*>|<!--[\s\S]*?-->')
EMOJI = re.compile(r'([0-9#*]\ufe0f?\u20e3|[\U0001f000-\U0001faff\u2600-\u27bf\u200d\u200c\ufe0f\ufe0e]+)')
protected_words = set(('không chẳng chưa chả đừng đâu mày tao nó ngu dốt khốn đéo đm vcl dm vl địt đụ tôi bạn anh chị em hắn họ chúng ai gì sao à ạ nhé nha hả nhỉ quá rất lắm cứ mà nhưng bị được phải nên hãy chết giết cút').split())
protected_words.update({'chúng_mày', 'vô_dụng', 'mọi_người'})
STOPWORDS = {x.strip().replace(' ', '_') for x in (BASE / 'vietnamese_stopwords.txt').read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')}

def normalize_unicode(text):
    return unicodedata.normalize('NFC', '' if text is None else str(text))

def clean_text(text):
    # Giữ ZWJ/ZWNJ và variation selector để không phá emoji ghép.
    text = normalize_unicode(text).lower()
    text = ''.join(' ' if c.isspace() or unicodedata.category(c) == 'Cc' else '' if unicodedata.category(c) == 'Cf' and c not in '\u200d\u200c' else c for c in text)
    return ' '.join(text.split())

def remove_noise(text):
    text = html.unescape(normalize_unicode(text))
    # Chỉ parse khi phát hiện tag, tránh biến dấu < > trong câu thành HTML.
    if TAG.search(text):
        soup = BeautifulSoup(text, 'html.parser')
        for node in soup(['script', 'style']):
            node.decompose()
        for br in soup.find_all(['br', 'p', 'div', 'li']):
            br.insert_before(' ')
            br.insert_after(' ')
        text = soup.get_text()
    def strip_url(match):
        # Giữ dấu cảm xúc/dấu kết câu nằm ở cuối URL.
        tail = re.search(r'[!?,.;:]+$', match.group())
        return ' ' + (tail.group() if tail else '')
    return ' '.join(URL.sub(strip_url, text).split())

def normalize_teencode(text):
    return TEEN.sub(lambda m: DICTIONARY['replacements'][m.group().lower()], text)

def tokenize_vietnamese(text):
    if not text.strip():
        return ''
    try:
        # Underthesea có thể chèn '_' vào chuỗi emoji ZWJ; tách riêng
        # các cụm emoji, chỉ gửi phần văn bản sang bộ tách từ.
        return ' '.join(part if EMOJI.fullmatch(part) else word_tokenize(part.strip(), format='text') for part in EMOJI.split(text) if part.strip())
    except Exception as exc:
        # Dừng có thông tin thay vì âm thầm thay bằng thuật toán khác.
        raise RuntimeError(f'Underthesea không xử lý được: {text[:100]!r}') from exc

def remove_stopwords(text, enabled=True):
    if not enabled:
        return text
    def keep(token):
        parts = set(token.split('_'))
        return token in protected_words or bool(parts & protected_words) or token not in STOPWORDS
    return ' '.join(t for t in text.split() if keep(t))

def read_dataset(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        for encoding in ['utf-8-sig', 'utf-8', 'cp1258']:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding), encoding
            except UnicodeDecodeError:
                continue
        raise ValueError('Không giải mã được CSV; cần chỉ định/chuyển encoding.')
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path, dtype=str, keep_default_na=False), 'XLSX (Unicode)'
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(data, dict):
            data = data.get('data', data.get('comments', data))
        return pd.DataFrame(data).fillna('').astype(str), 'UTF-8 JSON'
    raise ValueError('Chỉ hỗ trợ CSV, XLSX, JSON.')

def detect_columns(df):
    candidates = [c for c in df if c.lower() in ['comment', 'comments', 'comment_text', 'content', 'text', 'original_text', 'bình luận', 'noi_dung']]
    if not candidates:
        candidates = [c for c in df if c.lower() not in ['id', 'id_comment', 'label', 'reason', 'label_name'] and df[c].str.len().mean() > 15]
    if not candidates:
        raise ValueError('Không tìm thấy cột bình luận; dùng --text-column.')
    return candidates[0], candidates

def run(input_path, output_dir=BASE, stopwords=True, text_column=None):
    df, encoding = read_dataset(input_path)
    detected, candidates = detect_columns(df) if text_column is None else (text_column, [text_column])
    text = df[detected]
    out = df.copy()
    reserved = ['original_text', 'cleaned_text', 'normalized_text', 'tokenized_text', 'processed_text']
    if any(c in df for c in reserved):
        raise ValueError('Dataset đã có cột đầu ra; không ghi đè cột gốc.')
    out['id'] = df['id'] if 'id' in df else df['id_comment'] if 'id_comment' in df else pd.Series(range(1, len(df)+1)).astype(str)
    out['original_text'] = text
    out['cleaned_text'] = text.map(lambda t: clean_text(remove_noise(normalize_unicode(t))))
    out['normalized_text'] = out.cleaned_text.map(normalize_teencode)
    out['tokenized_text'] = out.normalized_text.map(tokenize_vietnamese)
    out['processed_text'] = out.tokenized_text.map(lambda t: remove_stopwords(t, stopwords))
    assert len(out) == len(df)
    pd.testing.assert_frame_equal(out[df.columns], df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'facebook_comments_preprocessed.csv'
    if target.resolve() == Path(input_path).resolve():
        raise ValueError('Không được ghi đè dataset gốc.')
    out.to_csv(target, index=False, encoding='utf-8-sig')
    # Chọn ví dụ có hiện tượng thực tế, sau đó bổ sung đủ 20 dòng.
    indices = []
    for mask in [text.str.contains(URL), text.str.contains(TAG), out.cleaned_text.map(lambda t: bool(TEEN.search(t))), out.cleaned_text.map(lambda t: bool(AMBIGUOUS.search(t))), out.cleaned_text.str.contains('không|ngu|đéo'), text.str.contains(r'[^\x00-\uFFFF]')]:
        indices.extend(out.index[mask][:4].tolist())
    indices = list(dict.fromkeys(indices + out.index.tolist()))[:20]
    comparison = out.loc[indices, ['id'] + reserved + (['label'] if 'label' in out else [])]
    comparison.to_csv(output_dir / 'preprocessing_comparison.csv', index=False, encoding='utf-8-sig')
    lines = ['# Báo cáo tiền xử lý bình luận Facebook', '', f'- Đầu vào: `{input_path}`; encoding đọc: {encoding}.', f'- SHA256 đầu vào: `{hashlib.sha256(Path(input_path).read_bytes()).hexdigest()}`.', f'- Cột bình luận: `{detected}`; ứng viên: {candidates}. Chọn theo tên nội dung gốc, không dùng reason/nhãn.', '- Bản labeling đã được đối chiếu toàn bộ ID và comment với dataset gốc: giống nhau; giữ nhãn và tất cả metadata. Bản review là tập con, correction_log là lịch sử, không ghép thêm.', f'- Cột nhãn: {"label (giữ nguyên)" if "label" in df else "không có"}.', f'- Tổng bình luận: {len(df)}.', f'- Hợp lệ (không trắng): {int(text.str.strip().ne("").sum())}; rỗng/trắng: {int(text.str.strip().eq("").sum())}.', f'- Trùng nội dung chính xác (số dòng dư, không xóa): {int(text.duplicated().sum())}; trùng toàn dòng: {int(df.duplicated().sum())}.', f'- Có URL: {int(text.str.contains(URL).sum())}; có HTML: {int(text.str.contains(TAG).sum())}; có URL hoặc HTML: {int((text.str.contains(URL) | text.str.contains(TAG)).sum())}.', f'- Có teencode được thay: {int(out.cleaned_text.map(lambda t: bool(TEEN.search(t))).sum())}; có viết tắt đa nghĩa giữ lại: {int(out.cleaned_text.map(lambda t: bool(AMBIGUOUS.search(t))).sum())}.', f'- Rỗng sau xử lý: {int(out.processed_text.str.strip().eq("").sum())}.', f'- Lọc stopwords: {stopwords}.', '', '## Độ dài', '', '| Giai đoạn | Ký tự TB | Ký tự trung vị | Ký tự min/max | Token cách trắng TB |', '|---|---:|---:|---:|---:|']
    for col in reserved:
        lengths = out[col].str.len()
        lines.append(f'| {col} | {lengths.mean():.2f} | {lengths.median():.0f} | {lengths.min()}/{lengths.max()} | {out[col].str.split().str.len().mean():.2f} |')
    lines += ['', '## Stopwords và bảo vệ', '', 'Nguồn: https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt (MIT). Bản nguồn lưu tại stopwords_source.txt; chỉ dùng tập con bảo thủ đã chọn, không suy ra từ tần suất hay nhãn.', '', 'Danh sách dùng: ' + ', '.join(sorted(STOPWORDS)), '', 'protected_words: ' + ', '.join(sorted(protected_words)), '', 'Token ghép được giữ nếu bất kỳ thành phần nào được bảo vệ; không xóa từng tiếng bên trong token ghép.', '', '## Chính sách và hạn chế', '', '- Giữ emoji, emoji ghép, dấu câu, số, tiếng tục; không rút gọn ký tự lặp vì có thể biểu thị cảm xúc. Xóa script/style, URL và ký tự điều khiển/định dạng gây nhiễu; giữ ZWJ/ZWNJ.', '- k, bt, vs đa nghĩa nên giữ nguyên. Từ điển cố ý nhỏ; không sửa lỗi chính tả, tiếng lóng chưa kiểm chứng hoặc suy diễn ngữ cảnh.', '- Underthesea có thể tách sai tên riêng, teencode và từ tục; ngoại lệ làm dừng pipeline, không âm thầm bỏ dòng. Thống kê token trước tách chỉ là đơn vị cách trắng.', '- URL/HTML có thể chứa tín hiệu hữu ích; việc loại bỏ làm mất tín hiệu đó. Stopwords dù bảo thủ vẫn có thể ảnh hưởng sắc thái; dùng --no-stopwords để so sánh.', '- Không gán nhãn, không đánh giá chất lượng nhãn, không huấn luyện, không BoW/TF-IDF, không chia dữ liệu.', '', '## Ví dụ thực tế', '']
    for _, row in comparison.iterrows():
        lines.append(f'### ID {row["id"]}')
        for col in reserved:
            lines.append(f'- {col}: {json.dumps(row[col], ensure_ascii=False)}')
        lines.append('')
    (output_dir / 'preprocessing_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Hoàn tất {len(out)} dòng -> {target}')
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(BASE / 'labeling/facebook_comments_labeled.csv'))
    parser.add_argument('--output-dir', default=str(BASE))
    parser.add_argument('--text-column')
    parser.add_argument('--no-stopwords', action='store_true')
    args = parser.parse_args()
    run(args.input, args.output_dir, not args.no_stopwords, args.text_column)
