# -*- coding: utf-8 -*-
"""
Vietnamese Toxic Comment Labeling System - Enhanced Version
============================================================
Multi-layer classification with improved context analysis for
Facebook comments about Vietnamese movie "Mưa Đỏ".

Methodology:
- Layer 1: Explicit toxic vocabulary (strong profanity, slurs, direct threats)
- Layer 2: Pattern-based insult detection (structural patterns)
- Layer 3: Disguised/teen-code profanity detection
- Layer 4: Context analysis (movie vs personal, quoting, narrative)
- Layer 5: Interpersonal aggression detection (mày/tao + insult combos)
- Layer 6: Confidence scoring and review flagging

Author: Nhật (preprocessing team member)
Date: 2026-11-09
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LabelResult:
    """Result of labeling a single comment."""
    label: Optional[int] = None
    label_name: str = ""
    confidence: float = 0.0
    needs_review: bool = False
    reason: str = ""
    matched_rules: list = field(default_factory=list)


# ============================================================================
# COMPREHENSIVE TOXIC PATTERNS
# ============================================================================

# === TIER 1: Explicit profanity (highest weight) ===
TIER1_PATTERNS = [
    # Core vulgar terms
    (r'\bl[oồ][nz]\b', 'vulgar_lon'),
    (r'\bcặc\b', 'vulgar_cac'),
    (r'\bđụ\b', 'vulgar_du'),
    (r'\bđ[ịi]t\b', 'vulgar_dit'),
    (r'\bđ[íi]t\s*m[eẹ]\b', 'vulgar_dit_me'),
    (r'\bđ[ií]t\s*cha\b', 'vulgar_dit_cha'),
    # Abbreviated profanity
    (r'(?<!\w)đm(?!\w)', 'abbrev_dm'),
    (r'\bđcm\b', 'abbrev_dcm'),
    (r'\bđkm\b', 'abbrev_dkm'),
    (r'\bđmm\b', 'abbrev_dmm'),
    (r'\bvkl\b', 'abbrev_vkl'),
    (r'\bdcm\b', 'abbrev_dcm2'),
    (r'\bdkm\b', 'abbrev_dkm2'),
    # Disguised profanity (l.ồ.n, c.ặ.c, etc.)
    (r'l[\.\s]*[oồ][\.\s]*[nz]', 'disguised_lon'),
    (r'c[\.\s]*[aặ][\.\s]*c', 'disguised_cac'),
    (r'đ[\.\s]*[ịi][\.\s]*t', 'disguised_dit'),
    # Variations: lol, loz, lồz (used as disguise for lồn)
    (r'\blol\b(?!\s*(liên\s*minh|game|LoL|League))', 'disguised_lol'),
    (r'\bloz\b', 'disguised_loz'),
    (r'\bxàm\s*l[oồ][nz]\b', 'xam_lon'),
    (r'\bxàm\s*lol\b', 'xam_lol'),
    (r'\bxàm\s*loz\b', 'xam_loz'),
    # "con cặc", "cái cc"
    (r'\bcon\s*cặc\b', 'con_cac'),
    (r'\bcái\s*cc\b', 'cai_cc'),
    (r'\bcon\s*c[aặ]c\b', 'con_cac2'),
    # "con đĩ", "con điếm"
    (r'\bcon\s*đĩ\b', 'con_di'),
    (r'\bcon\s*điếm\b', 'con_diem'),
    (r'\bthằng\s*đĩ\b', 'thang_di'),
    # "đ.m", "Đ.m"
    (r'[đĐ]\.m\b', 'dm_dotted'),
    # "méo" used as euphemism for "mẹ" in profanity context
    # Only when paired with other signals
]

# === TIER 2: Strong insults and attacks ===
TIER2_PATTERNS = [
    # Direct insults with subject
    (r'\bthằng\s*ngu\b', 'thang_ngu'),
    (r'\bcon\s*ngu\b', 'con_ngu'),
    (r'\bđồ\s*ngu\b', 'do_ngu'),
    (r'\bthằng\s*chó\b', 'thang_cho'),
    (r'\bcon\s*chó\b(?!\s*(cưng|yêu|nhỏ|đẹp|của|becgie|alaska|husky|corgi|poodle))', 'con_cho'),
    (r'\bthằng\s*điên\b', 'thang_dien'),
    (r'\bthằng\s*khùng\b', 'thang_khung'),
    (r'\bthằng\s*dốt\b', 'thang_dot'),
    (r'\bthằng\s*hèn\b', 'thang_hen'),
    (r'\bcon\s*điên\b(?!.*phim)', 'con_dien'),
    # Strong compound insults
    (r'\bmất\s*dạy\b', 'mat_day'),
    (r'\bvô\s*học\b', 'vo_hoc'),
    (r'\bvô\s*liêm\s*sỉ\b', 'vo_liem_si'),
    (r'\bkhốn\s*nạn\b', 'khon_nan'),
    (r'\bkhốn\s*kiếp\b', 'khon_kiep'),
    (r'\brác\s*rưởi\b', 'rac_ruoi'),
    (r'\bóc\s*chó\b', 'oc_cho'),
    (r'\bnão\s*chó\b', 'nao_cho'),
    (r'\btinh\s*trùng\s*khuyết\s*tật\b', 'tinh_trung_khuyet_tat'),
    # Threats
    (r'\bgiết\s*(mày|mẹ\s*mày|cha\s*mày|nó|chúng\s*mày|tụi\s*mày)\b', 'threat_giet'),
    (r'\bchết\s*(đi|mẹ|cha|cmn|cmnr|tiệt)\b', 'threat_chet'),
    (r'\bxuống\s*hố\b', 'threat_xuong_ho'),
    # Aggressive commands
    (r'\bcút\s*(đi|xéo|ngay|mẹ)\b', 'cmd_cut'),
    (r'\bbiến\s*(đi|ngay|mẹ)\b', 'cmd_bien'),
    (r'\bim\s*(mồm|miệng|cái\s*mồm|cái\s*miệng|đi)\b', 'cmd_im'),
    (r'\bcâm\s*(mồm|miệng|mõm|đi)\b', 'cmd_cam'),
    (r'\bcâm\s*mõm\b', 'cmd_cam_mom'),
    # Ancestor insults
    (r'\brác\s*mả\b', 'rac_ma'),
    (r'\bmả\s*(bố|mẹ|cha)\s*mày\b', 'ma_bo_may'),
    (r'\bông\s*nội.*\bngu\b', 'ong_noi_ngu'),
    (r'\bnhà\s*mày\b.*\brác\b', 'nha_may_rac'),
    (r'\b3?\s*đời\b.*\brác\b', 'ba_doi_rac'),
    (r'\bba\s*đời\b.*\brác\b', 'ba_doi_rac2'),
    # "xấc láo", "hỗn láo"
    (r'\bxấc\s*láo\b', 'xac_lao'),
    (r'\bhỗn\s*láo\b', 'hon_lao'),
    (r'\bxấc\s*xược\b', 'xac_xuoc'),
]

# === TIER 3: Moderate insults ===
TIER3_PATTERNS = [
    (r'\bngu\s*(lắm|quá|vl|vcl|vãi|vãi\s*chưởng|như\s*bò|như\s*chó|như\s*lợn|thế|vậy|thật|ghê|v[ck]l)\b', 'ngu_emphasis'),
    (r'\bdốt\s*(nát|quá|lắm|vl)\b', 'dot_emphasis'),
    (r'\bđần\s*(độn)?\b', 'dan_don'),
    (r'\bngớ\s*ngẩn\b', 'ngo_ngan'),
    (r'\bnhục\s*(nhã)?\b', 'nhuc'),
    (r'\bkinh\s*tởm\b', 'kinh_tom'),
    (r'\bghê\s*tởm\b', 'ghe_tom'),
    (r'\btởm\b(?!.*phim)', 'tom'),
    (r'\bđê\s*tiện\b', 'de_tien'),
    (r'\bbẩn\s*thỉu\b', 'ban_thiu'),
    (r'\bvô\s*dụng\b(?=.*(mày|bạn|ông|bà|nó|thằng|con))', 'vo_dung_personal'),
    (r'\bnão\s*(ngắn|trống|rỗng)\b', 'nao_ngan'),
    (r'\btrong\s*não\b.*\brác\b', 'trong_nao_rac'),
    (r'\bnão\b.*\brác\b', 'nao_rac'),
    # Contempt
    (r'\bkhinh\s*(bỉ|thường)\b', 'khinh'),
    (r'\bchó\s*chết\b', 'cho_chet'),
    # Derogatory labels
    (r'\bphản\s*động\b', 'phan_dong'),
    (r'\btay\s*sai\b(?!.*(phim|nhân\s*vật|vai|đóng))', 'tay_sai'),
    (r'\bba\s*que\b', 'ba_que'),
    (r'\bbò\s*đỏ\b', 'bo_do'),
    (r'\bdư\s*luận\s*viên\b', 'du_luan_vien'),
    # "như con" + animal (insult)
    (r'\bnhư\s*con\s*(chó|heo|lợn|bò|khỉ)\b', 'nhu_con_animal'),
    # Gendered insults
    (r'\bđàn\s*bà\b.*\b(ngu|dốt|nông\s*cạn)\b', 'gendered_insult'),
]

# === TIER 4: Interpersonal aggression patterns ===
TIER4_PATTERNS = [
    # "mày" + negative adjective (strong personal attack signal)
    (r'\bmày\b.*\b(ngu|dốt|điên|khùng|hèn|câm|điếc|mù|rác|đần|ngớ\s*ngẩn)\b', 'may_insult'),
    (r'\b(ngu|dốt|điên|khùng|hèn|câm|điếc|mù|rác|đần)\b.*\bmày\b', 'insult_may'),
    # "tao" + aggressive verbs toward "mày"
    (r'\btao\b.*\b(đánh|giết|đập|đấm|chém)\b.*\bmày\b', 'tao_violence_may'),
    (r'\btao\b.*\bmày\b.*\b(đánh|giết|đập|đấm|chém)\b', 'tao_may_violence'),
    # "loại" + pronoun (dehumanizing)
    (r'\bloại\s*(như\s*)?(mày|bạn|ông|bà|nó|tụi|chúng)\b', 'loai_pronoun'),
    (r'\b(cái\s*)?loại\b.*\b(ngu|dốt|rác|hèn)\b', 'loai_insult'),
    # Sarcastic/mocking "mày" combinations  
    (r'\bmày\s*(là|biết)\s*(cái?\s*)?(gì|gj|j|đếch|đéo|cc)\b', 'may_dismissive'),
    # "bố mày", "mẹ mày" used aggressively
    (r'\b(bố|cha)\s*mày\b', 'bo_may'),
    (r'\bmẹ\s*mày\b', 'me_may'),
    # Telling someone to go away rudely
    (r'\b(cút|biến|xéo|lượn)\b.*\b(đi|ngay|mau)\b', 'go_away_rude'),
]

# === TIER 5: Context-dependent words that need special handling ===
# These are NOT toxic by default - only toxic in certain contexts
CONTEXT_WORDS = {
    'đéo': {
        'base_pattern': r'\bđéo\b',
        'toxic_boost': [
            (r'\bmày\b.*\bđéo\b', 0.3),
            (r'\bđéo\b.*\bmày\b', 0.3),
            (r'\bđéo\s*biết\s*(cái\s*)?(gì|j)\b.*\b(mà|nói|bàn)\b', 0.2),
        ],
        'nontoxic_boost': [
            (r'\bsợ\s*(cái\s*)?đéo\b', 0.4),  # "sợ đéo gì" - bravado
            (r'\bđéo\s*sợ\b', 0.3),
            (r'\bt\s*đéo\b', 0.2),  # self-expression
            (r'\bđéo\s*(bao\s*giờ|có)\b', 0.2),
        ],
        'neutral_score': 0.0,  # by itself, not toxic
    },
    'vcl': {
        'base_pattern': r'\bvcl\b',
        'toxic_boost': [
            (r'\bngu\s*vcl\b', 0.4),
            (r'\bghét.*vcl\b', 0.2),
            (r'\bác\s*vcl\b', 0.15),
        ],
        'nontoxic_boost': [
            (r'\bhay\s*vcl\b', 0.4),
            (r'\bđẹp\s*vcl\b', 0.4),
            (r'\bngon\s*vcl\b', 0.3),
            (r'\bcảm\s*động\s*vcl\b', 0.4),
            (r'\bkhóc.*vcl\b', 0.3),
            (r'\bvui\s*vcl\b', 0.3),
            (r'\bbuồn\s*vcl\b', 0.2),
        ],
        'neutral_score': 0.0,
    },
}

# === Movie/narrative context indicators (reduce toxic score) ===
MOVIE_CONTEXT = [
    'phim', 'nhân vật', 'cảnh', 'diễn xuất', 'vai', 'đạo diễn', 'kịch bản',
    'review', 'trailer', 'rạp', 'chiếu', 'xem phim', 'bộ phim',
    'lịch sử', 'chiến tranh', 'thành cổ', 'quảng trị', 'mưa đỏ',
    'tiểu thuyết', 'truyện', 'tác phẩm', 'sản xuất',
    'trong phim', 'trong truyện', 'đoạn phim', 'cảnh phim',
    'diễn viên', 'thể loại', 'nội dung phim',
    'chiến sĩ', 'bộ đội', 'người lính', 'đồng đội', 'liệt sĩ',
]

# Character names that appear in movie discussions
CHARACTER_NAMES = ['sen', 'cường', 'tạ', 'quang', 'hồng', 'tú', 'tấn',
                   'anh tạ', 'ông tạ', 'a tạ']

NARRATIVE_INDICATORS = [
    'trong phim', 'trong truyện', 'nhân vật', 'vai diễn', 'cảnh này',
    'đoạn này', 'khúc này', 'phân cảnh', 'diễn viên', 'đóng vai',
    'hóa điên', 'hoá điên', 'giả điên', 'giả khùng', 'bị điên',
    'lời thoại', 'câu thoại', 'câu nói', 'trích dẫn',
]

# Words that are NOT insults in certain movie contexts
MOVIE_SAFE_WORDS = {
    'điên': ['hóa điên', 'hoá điên', 'giả điên', 'bị điên', 'nhân vật.*điên',
             'sen.*điên', 'điên.*sen', 'phim.*điên', 'chiến tranh.*điên'],
    'khùng': ['giả khùng', 'bị khùng', 'nhân vật.*khùng', 'sen.*khùng', 'khùng.*sen'],
    'chó': ['sợ cái chó gì', 'sợ chó gì', 'cái chó gì', 'trộm chó', 'bắt chó',
            'chó nghiệp vụ', 'con chó cưng'],
    'hèn': ['vai.*hèn', 'nhân vật.*hèn', 'hèn.*vai', 'quang.*hèn', 'hèn.*quang'],
    'chết': ['chết.*phim', 'hy sinh', 'ngã xuống', 'chết mẹ sợ', 'chết mẹ rồi',
             'khóc chết', 'chết chóc', 'đẹp chết'],
    'ngu': ['ngu ngốc.*chiến tranh', 'ngu dốt.*chiến tranh'],
}


class EnhancedToxicClassifier:
    """Enhanced multi-layer Vietnamese toxic comment classifier."""
    
    def __init__(self):
        """Initialize and compile all patterns."""
        self.tier1 = [(re.compile(p, re.IGNORECASE), name) for p, name in TIER1_PATTERNS]
        self.tier2 = [(re.compile(p, re.IGNORECASE), name) for p, name in TIER2_PATTERNS]
        self.tier3 = [(re.compile(p, re.IGNORECASE), name) for p, name in TIER3_PATTERNS]
        self.tier4 = [(re.compile(p, re.IGNORECASE), name) for p, name in TIER4_PATTERNS]
        
        # Context words
        self.ctx_words = {}
        for word, config in CONTEXT_WORDS.items():
            self.ctx_words[word] = {
                'base': re.compile(config['base_pattern'], re.IGNORECASE),
                'toxic': [(re.compile(p, re.IGNORECASE), s) for p, s in config['toxic_boost']],
                'nontoxic': [(re.compile(p, re.IGNORECASE), s) for p, s in config['nontoxic_boost']],
            }
        
        # Movie safe words
        self.movie_safe = {}
        for word, patterns in MOVIE_SAFE_WORDS.items():
            self.movie_safe[word] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        self.movie_ctx_lower = [w.lower() for w in MOVIE_CONTEXT]
        self.char_names_lower = [w.lower() for w in CHARACTER_NAMES]
        self.narrative_lower = [w.lower() for w in NARRATIVE_INDICATORS]
    
    def _count_movie_context(self, text_lower: str) -> int:
        """Count movie context indicators."""
        return sum(1 for w in self.movie_ctx_lower if w in text_lower)
    
    def _has_character_ref(self, text_lower: str) -> bool:
        """Check if text references movie characters."""
        return any(w in text_lower for w in self.char_names_lower)
    
    def _has_narrative(self, text_lower: str) -> bool:
        """Check for narrative/plot discussion."""
        return any(w in text_lower for w in self.narrative_lower)
    
    def _is_movie_safe(self, text: str, word: str) -> bool:
        """Check if a word is used in a movie-safe context."""
        if word in self.movie_safe:
            return any(p.search(text) for p in self.movie_safe[word])
        return False
    
    def _has_personal_pronouns(self, text_lower: str) -> bool:
        """Check for personal attack pronouns (mày/tao used aggressively)."""
        has_may = bool(re.search(r'\bmày\b', text_lower))
        has_tao = bool(re.search(r'\btao\b', text_lower))
        return has_may or has_tao
    
    def _check_aggressive_emoji(self, text: str) -> float:
        """Check for aggressive emoji patterns."""
        score = 0.0
        angry_emoji = ['🤬', '😡', '💢', '🖕', '👊']
        for e in angry_emoji:
            if e in text:
                score += 0.05
        return min(score, 0.15)
    
    def _is_attacking_commenter(self, text_lower: str) -> bool:
        """Detect if comment is attacking another commenter (not a movie character)."""
        # Patterns that indicate replying to / attacking another commenter
        commenter_attack = [
            r'\bbạn\s*(là|bị|có)\s*(đồ|thằng|con|loại|cái)',
            r'\b(ông|bà)\s*(này|kia|đó)\b.*\b(ngu|dốt|rác|hèn)',
            r'\bnhà\s*(mày|bạn|ông|bà)\b',
            r'\bmày\b.*\bnói\b', r'\bnói\b.*\bmày\b',
            r'\bđọc\b.*\b(lại|kỹ)\b.*\b(đi|nha|nhé)\b',
            r'\bsao\s*mà\b.*\bngu\b',
        ]
        return any(re.search(p, text_lower) for p in commenter_attack)
    
    def classify(self, comment_text: str) -> LabelResult:
        """
        Classify a comment as Toxic (1) or Non-toxic (0).
        
        Multi-layer analysis:
        1. Pattern matching across 4 tiers
        2. Context analysis (movie vs personal)
        3. Context-dependent word analysis
        4. Score aggregation and decision
        """
        result = LabelResult()
        
        # Handle empty/invalid input
        if not comment_text or not isinstance(comment_text, str):
            result.label = 0
            result.label_name = "Non-toxic"
            result.confidence = 0.5
            result.needs_review = True
            result.reason = "Bình luận trống hoặc không hợp lệ"
            return result
        
        text = comment_text.strip()
        if len(text) == 0:
            result.label = 0
            result.label_name = "Non-toxic"
            result.confidence = 0.5
            result.needs_review = True
            result.reason = "Bình luận trống"
            return result
        
        text_lower = text.lower()
        
        # Emoji-only comments
        if len(text) <= 5 and not re.search(r'[a-zA-ZÀ-ỹ]', text):
            result.label = 0
            result.label_name = "Non-toxic"
            result.confidence = 0.8
            result.reason = "Chỉ chứa emoji/ký tự đặc biệt"
            return result
        
        # ========== PATTERN MATCHING ==========
        toxic_score = 0.0
        matched_rules = []
        reasons = []
        
        # Tier 1: Explicit profanity (weight: 0.7)
        tier1_hits = []
        for pattern, name in self.tier1:
            if pattern.search(text):
                tier1_hits.append(name)
        if tier1_hits:
            toxic_score += 0.7
            matched_rules.extend(tier1_hits)
            reasons.append("Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng")
        
        # Tier 2: Strong insults (weight: 0.55)
        tier2_hits = []
        for pattern, name in self.tier2:
            if pattern.search(text):
                tier2_hits.append(name)
        if tier2_hits:
            toxic_score += 0.55
            matched_rules.extend(tier2_hits)
            reasons.append("Chứa lăng mạ/đe dọa trực tiếp")
        
        # Tier 3: Moderate insults (weight: 0.35)
        tier3_hits = []
        for pattern, name in self.tier3:
            if pattern.search(text):
                tier3_hits.append(name)
        if tier3_hits:
            toxic_score += 0.35
            matched_rules.extend(tier3_hits)
            reasons.append("Chứa từ ngữ xúc phạm mức trung bình")
        
        # Tier 4: Interpersonal aggression (weight: 0.45)
        tier4_hits = []
        for pattern, name in self.tier4:
            if pattern.search(text):
                tier4_hits.append(name)
        if tier4_hits:
            toxic_score += 0.45
            matched_rules.extend(tier4_hits)
            reasons.append("Phát hiện tấn công cá nhân")
        
        # ========== CONTEXT-DEPENDENT WORDS ==========
        for word, config in self.ctx_words.items():
            if config['base'].search(text):
                word_toxic = 0.0
                word_nontoxic = 0.0
                for pat, score in config['toxic']:
                    if pat.search(text):
                        word_toxic += score
                for pat, score in config['nontoxic']:
                    if pat.search(text):
                        word_nontoxic += score
                
                if word_toxic > word_nontoxic:
                    toxic_score += min(word_toxic, 0.3)
                    matched_rules.append(f'ctx_{word}_toxic')
                elif word_nontoxic > word_toxic:
                    # Has non-toxic usage - slightly reduce overall score
                    toxic_score = max(0, toxic_score - 0.1)
                    matched_rules.append(f'ctx_{word}_nontoxic')
                else:
                    # Ambiguous - small boost if other signals exist
                    if toxic_score > 0:
                        toxic_score += 0.05
        
        # ========== CONTEXT MODIFIERS ==========
        movie_ctx_count = self._count_movie_context(text_lower)
        has_char = self._has_character_ref(text_lower)
        has_narr = self._has_narrative(text_lower)
        has_personal = self._has_personal_pronouns(text_lower)
        is_attacking = self._is_attacking_commenter(text_lower)
        
        # Check if insult words are in movie-safe context
        safe_word_found = False
        for word in ['điên', 'khùng', 'chó', 'hèn', 'chết', 'ngu']:
            if word in text_lower and self._is_movie_safe(text, word):
                safe_word_found = True
                break
        
        # Apply context modifiers
        if toxic_score > 0:
            # If it's clearly about the movie and NOT personal attack
            if movie_ctx_count >= 2 and not has_personal and not is_attacking:
                if tier1_hits:
                    # Even with movie context, explicit profanity is still flagged
                    toxic_score *= 0.7
                    result.needs_review = True
                    reasons.append("[Ngữ cảnh phim nhưng có từ mạnh]")
                else:
                    toxic_score *= 0.3
                    reasons.append("[Giảm điểm: ngữ cảnh thảo luận phim]")
            
            if has_narr and not has_personal:
                toxic_score *= 0.5
                reasons.append("[Giảm điểm: kể chuyện/phân tích phim]")
            
            if safe_word_found and not has_personal:
                toxic_score *= 0.4
                reasons.append("[Giảm điểm: từ nhạy cảm trong ngữ cảnh an toàn]")
            
            # If HAS personal attack markers, boost score
            if has_personal and (tier1_hits or tier2_hits):
                toxic_score += 0.15
                reasons.append("Kết hợp mày/tao với từ xúc phạm")
            
            if is_attacking:
                toxic_score += 0.1
                reasons.append("Tấn công người bình luận khác")
        
        # Aggressive emoji bonus
        emoji_score = self._check_aggressive_emoji(text)
        if emoji_score > 0 and toxic_score > 0:
            toxic_score += emoji_score
        
        # ========== SPECIAL CASES ==========
        
        # "phim rác" - criticizing the movie, borderline
        # Not toxic unless combined with personal attack
        if re.search(r'\bphim\s*(rác|loz?l?|cứt)\b', text_lower):
            if not has_personal and not is_attacking and not tier1_hits:
                # Just criticizing the movie = non-toxic (harsh but not personal attack)
                toxic_score = max(toxic_score * 0.5, 0.15)
                reasons.append("[Chê phim, không tấn công cá nhân]")
                result.needs_review = True
            else:
                # Criticizing movie AND attacking someone
                toxic_score += 0.1
        
        # "rác phẩm" - harsh movie criticism
        if re.search(r'\brác\s*phẩm\b', text_lower):
            if not has_personal and not is_attacking:
                toxic_score = max(toxic_score * 0.4, 0.1)
                reasons.append("[Chê phim là 'rác phẩm', không tấn công cá nhân]")
                result.needs_review = True
        
        # Comments about historical enemies (not toxic toward other commenters)
        historical_context = re.search(
            r'\b(xâm\s*lược|quân\s*(địch|thù)|bọn\s*(xâm\s*lược|giặc)|kẻ\s*thù|ngoại\s*bang)\b',
            text_lower
        )
        if historical_context and not has_personal and not is_attacking:
            toxic_score *= 0.3
            reasons.append("[Ngữ cảnh lịch sử/chiến tranh]")
        
        # Quoting movie dialogue (e.g., "sợ cái chó gì")
        if re.search(r'(câu\s*thoại|lời\s*thoại|nói\s*rằng|tóm\s*tắt)', text_lower):
            if not is_attacking:
                toxic_score *= 0.3
                reasons.append("[Trích dẫn lời thoại phim]")
        
        # ========== FINAL DECISION ==========
        
        result.matched_rules = matched_rules
        
        if toxic_score >= 0.45:
            result.label = 1
            result.label_name = "Toxic"
            result.confidence = min(0.95, 0.5 + toxic_score * 0.25)
            result.reason = "; ".join(reasons) if reasons else "Phát hiện ngôn ngữ độc hại"
            
            if result.confidence < 0.65:
                result.needs_review = True
                
        elif toxic_score >= 0.25:
            # Borderline
            result.label = 1
            result.label_name = "Toxic"
            result.confidence = 0.4 + toxic_score * 0.3
            result.needs_review = True
            result.reason = ("; ".join(reasons) + " [Trường hợp biên]") if reasons else "Dấu hiệu không rõ ràng"
            
        else:
            result.label = 0
            result.label_name = "Non-toxic"
            
            if matched_rules:
                # Had some matches but context overrode them
                result.confidence = 0.6
                result.needs_review = True
                result.reason = "Có từ nhạy cảm nhưng trong ngữ cảnh phù hợp"
            else:
                result.confidence = 0.85
                result.reason = "Không phát hiện dấu hiệu ngôn ngữ độc hại"
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_labeling(input_path: str, output_dir: str):
    """Run the labeling pipeline on the dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Reading dataset: %s", input_path)
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info("Shape: %s, Columns: %s", df.shape, df.columns.tolist())
    
    comment_col = 'comment'
    id_col = 'id_comment'
    
    classifier = EnhancedToxicClassifier()
    
    logger.info("Classifying %d comments...", len(df))
    
    labels = []
    label_names = []
    confidences = []
    reviews = []
    reasons_list = []
    
    for idx, row in df.iterrows():
        text = str(row[comment_col]) if pd.notna(row[comment_col]) else ""
        result = classifier.classify(text)
        
        labels.append(result.label)
        label_names.append(result.label_name)
        confidences.append(round(result.confidence, 3))
        reviews.append(result.needs_review)
        reasons_list.append(result.reason)
        
        if (idx + 1) % 5000 == 0:
            logger.info("  Processed %d/%d", idx + 1, len(df))
    
    logger.info("Classification complete.")
    
    # Build output
    df_out = df.copy()
    if 'label' in df_out.columns:
        if df_out['label'].notna().any():
            df_out = df_out.rename(columns={'label': 'label_original'})
        else:
            df_out = df_out.drop(columns=['label'])
    
    df_out['label'] = labels
    df_out['label_name'] = label_names
    df_out['confidence'] = confidences
    df_out['needs_review'] = reviews
    df_out['reason'] = reasons_list
    
    # Save files
    labeled_path = output_dir / 'facebook_comments_labeled.csv'
    df_out.to_csv(labeled_path, index=False, encoding='utf-8-sig')
    logger.info("Saved: %s", labeled_path)
    
    review_df = df_out[df_out['needs_review'] == True].copy()
    review_path = output_dir / 'facebook_comments_review.csv'
    review_df.to_csv(review_path, index=False, encoding='utf-8-sig')
    logger.info("Saved review file: %s (%d comments)", review_path, len(review_df))
    
    # Stats
    total = len(df_out)
    toxic_n = int((df_out['label'] == 1).sum())
    nontoxic_n = int((df_out['label'] == 0).sum())
    undet = int(df_out['label'].isna().sum())
    review_n = int(df_out['needs_review'].sum())
    
    toxic_pct = round(toxic_n / total * 100, 2) if total else 0
    nontoxic_pct = round(nontoxic_n / total * 100, 2) if total else 0
    avg_conf = round(df_out['confidence'].mean(), 3)
    
    # Confidence distribution
    toxic_high_conf = int(((df_out['label'] == 1) & (df_out['confidence'] >= 0.7)).sum())
    toxic_low_conf = int(((df_out['label'] == 1) & (df_out['confidence'] < 0.7)).sum())
    
    stats = {
        'total': total, 'toxic': toxic_n, 'nontoxic': nontoxic_n,
        'undetermined': undet, 'review_needed': review_n,
        'toxic_pct': toxic_pct, 'nontoxic_pct': nontoxic_pct,
        'avg_confidence': avg_conf,
        'toxic_high_conf': toxic_high_conf, 'toxic_low_conf': toxic_low_conf,
    }
    
    logger.info("="*60)
    logger.info("RESULTS:")
    logger.info("  Total: %d", total)
    logger.info("  Toxic: %d (%.2f%%)", toxic_n, toxic_pct)
    logger.info("  Non-toxic: %d (%.2f%%)", nontoxic_n, nontoxic_pct)
    logger.info("  Undetermined: %d", undet)
    logger.info("  Needs review: %d", review_n)
    logger.info("  Avg confidence: %.3f", avg_conf)
    logger.info("  Toxic (high conf >=0.7): %d", toxic_high_conf)
    logger.info("  Toxic (low conf <0.7): %d", toxic_low_conf)
    
    # Generate report
    _generate_report(output_dir, stats, df_out)
    
    # Print sample toxic comments for verification
    print("\n=== SAMPLE TOXIC COMMENTS ===")
    toxic_df = df_out[df_out['label'] == 1].sort_values('confidence', ascending=False)
    for _, row in toxic_df.head(20).iterrows():
        cmt = str(row['comment'])[:120]
        print("[%d] (conf=%.2f) %s" % (row['id_comment'], row['confidence'], cmt))
    
    print("\n=== SAMPLE BORDERLINE (needs review) ===")
    border = df_out[(df_out['needs_review'] == True)].sort_values('confidence')
    for _, row in border.head(15).iterrows():
        cmt = str(row['comment'])[:120]
        print("[%d] (label=%s, conf=%.2f) %s" % (row['id_comment'], row['label_name'], row['confidence'], cmt))
    
    return df_out, stats


def _generate_report(output_dir: Path, stats: dict, df: pd.DataFrame):
    """Generate labeling_report.md."""
    
    toxic_samples = df[df['label'] == 1].sort_values('confidence', ascending=False).head(15)
    review_samples = df[df['needs_review'] == True].sort_values('confidence').head(15)
    
    conf_dist = df.groupby(pd.cut(df['confidence'], bins=[0, 0.4, 0.6, 0.8, 1.0])).size()
    
    report = f"""# Báo Cáo Gán Nhãn Toxic / Non-toxic

## Dataset Bình Luận Facebook – Phim "Mưa Đỏ"

**Ngày thực hiện:** 2026-11-09  
**Phương pháp:** Hệ thống phân loại đa tầng dựa trên quy tắc ngữ nghĩa (AI-assisted rule-based)  
**Dataset gốc:** `mua_do_comments_dedup.csv`

---

## 1. Tổng Quan Dataset Đầu Vào

| Thông tin | Giá trị |
|---|---|
| Tổng số bình luận | {stats['total']:,} |
| Cột nội dung | `comment` |
| Cột ID | `id_comment` |
| Encoding | UTF-8 with BOM |
| Bình luận trống | 0 |
| Bình luận trùng lặp | 0 |
| Nhãn ban đầu | Không có (toàn bộ NaN) |

## 2. Kết Quả Phân Loại

| Nhãn | Số lượng | Tỷ lệ |
|---|---|---|
| **Non-toxic (0)** | {stats['nontoxic']:,} | {stats['nontoxic_pct']}% |
| **Toxic (1)** | {stats['toxic']:,} | {stats['toxic_pct']}% |
| Chưa xác định | {stats['undetermined']} | - |

**Cần kiểm tra lại (needs_review = TRUE):** {stats['review_needed']:,} bình luận ({round(stats['review_needed']/stats['total']*100, 1)}%)

### Phân bố Toxic theo độ tin cậy

| Loại | Số lượng |
|---|---|
| Toxic có độ tin cậy cao (≥ 0.7) | {stats['toxic_high_conf']} |
| Toxic có độ tin cậy thấp (< 0.7) | {stats['toxic_low_conf']} |

**Độ tin cậy trung bình toàn bộ:** {stats['avg_confidence']}

## 3. Tiêu Chí Phân Loại Chi Tiết

### 3.1. Toxic (label = 1)
Bình luận bị phân loại Toxic khi chứa một hoặc nhiều dấu hiệu:
- **Tầng 1 (trọng số cao nhất):** Từ ngữ tục tĩu rõ ràng, xúc phạm nghiêm trọng, và các biến thể viết tắt/ngụy trang.
- **Tầng 2:** Lăng mạ trực tiếp ("thằng ngu", "con chó"), đe dọa, lệnh im miệng/cút đi.
- **Tầng 3:** Xúc phạm mức trung bình ("đần độn", "khinh bỉ"), nhãn miệt thị ("phản động", "tay sai").
- **Tầng 4:** Tấn công cá nhân kết hợp đại từ "mày/tao" với từ xúc phạm.

### 3.2. Non-toxic (label = 0)
- Bình luận thông thường, khen/chê phim, thảo luận lịch sử.
- Phản biện gay gắt nhưng không công kích cá nhân.
- Bình luận có cảm xúc tiêu cực nhưng không hướng tới xúc phạm ai.

### 3.3. Xử Lý Ngữ Cảnh
- Từ "điên", "khùng" trong ngữ cảnh thảo luận nhân vật Sen → **Non-toxic**.
- "Sợ cái chó gì" (lời thoại phim) → **Non-toxic**.
- "Phim rác" (chê phim, không tấn công người) → **Non-toxic** hoặc **cần review**.
- "vcl" dùng để nhấn mạnh tích cực ("hay vcl", "cảm động vcl") → **Non-toxic**.
- "đéo" dùng để nhấn mạnh ("sợ đéo gì") → xét ngữ cảnh.

## 4. Phương Pháp Kỹ Thuật

### 4.1. Kiến trúc
Hệ thống phân loại đa tầng (5 tầng):
1. **Phát hiện từ vựng:** Regex matching với 70+ mẫu từ vựng toxic tiếng Việt, bao gồm biến thể teen code và ngụy trang.
2. **Phân tích cấu trúc:** Nhận diện các cấu trúc câu xúc phạm (chủ ngữ + từ xúc phạm).
3. **Phân tích ngữ cảnh:** Xác định ngữ cảnh thảo luận phim, kể chuyện, trích dẫn.
4. **Phân tích tấn công cá nhân:** Nhận diện khi bình luận nhắm vào người bình luận khác.
5. **Tính điểm và ra quyết định:** Kết hợp tất cả tín hiệu với trọng số phù hợp.

### 4.2. Công cụ
- Python 3 + pandas + regex
- Hệ thống rule-based đa tầng với phân tích ngữ cảnh
- **Không** phải keyword matching đơn thuần: có phân tích ngữ cảnh, phân biệt tấn công cá nhân vs thảo luận phim

### 4.3. Giới hạn phương pháp
- Không thể hiểu sarcasm/mỉa mai phức tạp
- Không phân tích được ngữ cảnh ngoài văn bản (ai nói với ai, thread nào)
- Có thể bỏ sót toxic tinh vi hoặc teen code chưa có trong từ điển

## 5. Mẫu Bình Luận Toxic (Top theo Confidence)

| ID | Bình luận (rút gọn) | Conf. | Lý do |
|---|---|---|---|
"""
    for _, row in toxic_samples.iterrows():
        cmt = str(row['comment'])[:80].replace('|', '/').replace('\n', ' ')
        reason = str(row['reason'])[:60].replace('|', '/').replace('\n', ' ')
        report += f"| {row['id_comment']} | {cmt} | {row['confidence']} | {reason} |\n"

    report += f"""

## 6. Mẫu Bình Luận Cần Kiểm Tra

| ID | Bình luận (rút gọn) | Label | Conf. |
|---|---|---|---|
"""
    for _, row in review_samples.iterrows():
        cmt = str(row['comment'])[:80].replace('|', '/').replace('\n', ' ')
        report += f"| {row['id_comment']} | {cmt} | {row['label_name']} | {row['confidence']} |\n"
    
    report += f"""

## 7. Hạn Chế và Rủi Ro

> **⚠️ QUAN TRỌNG:** Nhãn được tạo bởi hệ thống heuristic, chưa được con người xác nhận. Không nên coi đây là ground truth mà không qua bước kiểm tra thủ công.

1. **False negative (bỏ sót toxic):** Bình luận toxic tinh vi, mỉa mai, công kích gián tiếp có thể bị phân loại nhầm là Non-toxic.
2. **False positive (gán nhầm toxic):** Bình luận có từ nhạy cảm nhưng dùng trong ngữ cảnh phim/lịch sử có thể bị gán nhầm là Toxic.
3. **Thiên lệch ngữ cảnh:** Dataset chủ yếu về phim "Mưa Đỏ" nên classifier được điều chỉnh cho ngữ cảnh này; có thể không phù hợp với dataset khác.
4. **Teen code chưa đầy đủ:** Một số biến thể viết tắt/lệch chưa được bao phủ.
5. **Thiếu ngữ cảnh thread:** Không biết ai trả lời ai, nên khó phân biệt trích dẫn vs tấn công.

## 8. Khuyến Nghị

1. **Bắt buộc kiểm tra:** Tất cả bình luận `needs_review = TRUE` ({stats['review_needed']:,} bình luận).
2. **Ưu tiên kiểm tra:** Toxic có `confidence < 0.7` ({stats['toxic_low_conf']} bình luận).
3. **Rà soát ngẫu nhiên:** 5-10% bình luận Non-toxic để phát hiện false negative.
4. **Sau khi kiểm tra:** Cập nhật cột `label` trong `facebook_comments_labeled.csv`.

## 9. Danh Sách File Đầu Ra

| File | Mô tả |
|---|---|
| `facebook_comments_labeled.csv` | Toàn bộ {stats['total']:,} bình luận với nhãn đầy đủ |
| `facebook_comments_review.csv` | {stats['review_needed']:,} bình luận cần kiểm tra lại |
| `labeling_report.md` | Báo cáo này |
| `label_toxic_v2.py` | Mã nguồn classifier |
"""
    
    report_path = output_dir / 'labeling_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info("Report saved: %s", report_path)


if __name__ == '__main__':
    input_file = r'd:\btl kpdl\mua_do_comments_dedup.csv'
    output_directory = r'd:\btl kpdl\labeling'
    
    df_result, statistics = run_labeling(input_file, output_directory)

