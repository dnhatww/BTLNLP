"""Kiểm thử quy tắc và tính toàn vẹn dữ liệu thật."""
import tempfile
import unittest
from unittest.mock import patch
import unicodedata
from pathlib import Path
import preprocess as p

class PreprocessingTests(unittest.TestCase):
    def test_nfc(self):
        text = 'Tiếng Việt đậm đà'
        self.assertEqual(p.normalize_unicode(unicodedata.normalize('NFD', text)), text)

    def test_clean(self):
        self.assertEqual(p.clean_text('   ĐỒ   NGU!!!\tMày\nBIẾT GÌ???  '), 'đồ ngu!!! mày biết gì???')
        self.assertEqual(p.clean_text('a\u200bb\ufeffc\x00d'), 'abc d')

    def test_noise(self):
        self.assertEqual(p.clean_text(p.remove_noise('<p>Thằng này ngu thật!!!</p> https://example.com')), 'thằng này ngu thật!!!')
        self.assertEqual(p.remove_noise('A &amp; B &quot;C&quot;'), 'A & B "C"')
        self.assertEqual(p.remove_noise('<script>alert(1)</script><p>a</p><p>b</p>'), 'a b')
        self.assertEqual(p.remove_noise('a www.example.com!!!'), 'a !!!')
        self.assertEqual(p.remove_noise('1 < 2 > 0'), '1 < 2 > 0')
        self.assertEqual(p.remove_noise('&lt;p&gt;ngu&lt;/p&gt;'), 'ngu')

    def test_teencode_boundaries(self):
        self.assertEqual(p.normalize_teencode('mày ko bt j mà nói, ngu vcl'), 'mày không bt gì mà nói, ngu vcl')
        self.assertEqual(p.normalize_teencode('joker mikado k bt vs đéo đm vcl'), 'joker mikado k bt vs đéo đm vcl')

    def test_protected(self):
        old = p.STOPWORDS.copy()
        try:
            tokens = 'không chẳng chưa chả đừng đâu mày tao nó chúng_mày ngu dốt khốn vô_dụng đéo đm vcl không_biết'
            p.STOPWORDS.update(tokens.split())
            self.assertEqual(p.remove_stopwords(tokens), tokens)
        finally:
            p.STOPWORDS.clear()
            p.STOPWORDS.update(old)
        self.assertEqual(p.remove_stopwords('và mày không ngu'), 'mày không ngu')
        self.assertEqual(p.remove_stopwords('và mày', False), 'và mày')

    def test_underthesea(self):
        self.assertEqual(p.tokenize_vietnamese('học sinh này rất thông minh'), 'học_sinh này rất thông_minh')
        text = 'mày không ngu vcl!!! 🥹 👨‍👩‍👧‍👦'
        result = p.tokenize_vietnamese(text)
        for fragment in ['mày', 'không', 'ngu', 'vcl', '!', '🥹', '👨‍👩‍👧‍👦']:
            self.assertIn(fragment, result)
        self.assertIn('👨‍👩‍👧‍👦', p.clean_text(text))

    def test_tokenizer_exception(self):
        with patch.object(p, 'word_tokenize', side_effect=ValueError('lỗi kiểm thử')):
            with self.assertRaisesRegex(RuntimeError, 'Underthesea'):
                p.tokenize_vietnamese('bình luận')

    def test_empty(self):
        for fn in [p.normalize_unicode, p.remove_noise, p.clean_text, p.normalize_teencode, p.tokenize_vietnamese, p.remove_stopwords]:
            self.assertEqual(fn(''), '')

    def test_real_output_integrity(self):
        source, _ = p.read_dataset(p.BASE / 'labeling/facebook_comments_labeled.csv')
        result, _ = p.read_dataset(p.BASE / 'facebook_comments_preprocessed.csv')
        p.pd.testing.assert_frame_equal(source, result[source.columns])
        self.assertEqual(result.original_text.tolist(), source.comment.tolist())
        self.assertEqual(result.id.tolist(), source.id_comment.tolist())
        self.assertEqual((p.BASE / 'facebook_comments_preprocessed.csv').read_bytes()[:3], b'\xef\xbb\xbf')

    def test_read_formats(self):
        real, _ = p.read_dataset(p.BASE / 'mua_do_comments_dedup.csv')
        sample = real.head(3)
        with tempfile.TemporaryDirectory(dir=p.BASE) as directory:
            for extension in ['xlsx', 'json']:
                path = Path(directory) / ('sample.' + extension)
                if extension == 'xlsx':
                    sample.to_excel(path, index=False)
                else:
                    sample.to_json(path, orient='records', force_ascii=False)
                loaded, _ = p.read_dataset(path)
                p.pd.testing.assert_frame_equal(sample, loaded)

if __name__ == '__main__':
    unittest.main(verbosity=2)
