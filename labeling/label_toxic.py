# -*- coding: utf-8 -*-
"""
Vietnamese Toxic Comment Labeling System
=========================================
Multi-layer classification system for Facebook comments.

Methodology:
- Layer 1: Explicit toxic vocabulary detection (strong profanity, slurs, direct threats)
- Layer 2: Pattern-based toxic detection (insult patterns, aggressive structures)
- Layer 3: Context analysis (movie discussion vs personal attack, negation, quoting)
- Layer 4: Confidence scoring and review flagging

This is a heuristic-based system with contextual analysis, NOT a simple keyword matcher.
Limitations are documented in the output report.

Author: Nhật (preprocessing team member)
Date: 2026-11-09
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LabelResult:
    """Result of labeling a single comment."""
    label: Optional[int] = None        # 0=Non-toxic, 1=Toxic, None=undetermined
    label_name: str = ""                # "Toxic", "Non-toxic", or ""
    confidence: float = 0.0            # 0.0 to 1.0
    needs_review: bool = False
    reason: str = ""
    toxic_signals: list = field(default_factory=list)
    nontoxic_signals: list = field(default_factory=list)


# ============================================================================
# TOXIC VOCABULARY TIERS
# ============================================================================

# Tier 1: Very strong profanity / slurs - almost always toxic when directed at someone
TIER1_EXPLICIT = [
    # Vulgar sexual terms used as insults
    r'\bl[oồ]n\b', r'\bcặc\b', r'\bđụ\b', r'\bđ[ịi]t\b',
    r'\bđ[íi]t\s*mẹ\b', r'\bđ[íi]t\s*cha\b',
    # Common abbreviations of profanity
    r'\bđcm\b', r'\bđkm\b', r'\bđmm\b', r'\bvkl\b',
    r'\bdcm\b', r'\bdkm\b',
    # "đm" as standalone (not part of another word)
    r'(?<!\w)đm(?!\w)',
    # Extreme derogatory combinations
    r'\bcái\s*l[oồ]n\b', r'\bcon\s*đĩ\b', r'\bthằng\s*đĩ\b',
    r'\bcon\s*điếm\b',
]

# Tier 2: Strong insults and attacks
TIER2_STRONG_INSULTS = [
    # Direct personal insults
    r'\bthằng\s*ngu\b', r'\bcon\s*ngu\b', r'\bđồ\s*ngu\b',
    r'\bmất\s*dạy\b', r'\bvô\s*học\b', r'\bvô\s*liêm\s*sỉ\b',
    r'\bkhốn\s*nạn\b', r'\brác\s*rưởi\b',
    r'\bóc\s*chó\b', r'\bnão\s*chó\b',
    # Animal-based insults directed at people
    r'\bthằng\s*chó\b', r'\bcon\s*chó\b(?!\s*(cưng|yêu|nhỏ|đẹp|của))',
    r'\bnhư\s*(con\s*)?chó\b(?!\s*(cưng|trung\s*thành))',
    r'\bcon\s*lợn\b', r'\bcon\s*heo\b',
    # Threats
    r'\bgiết\s*(mày|mẹ\s*mày|cha\s*mày|nó|chúng\s*mày)\b',
    r'\bchết\s*(đi|mẹ|cha|cmn)\b',
    # Commands to shut up / go away aggressively
    r'\bcút\s*(đi|xéo|ngay)\b', r'\bbiến\s*đi\b',
    r'\bim\s*(mồm|miệng|cái\s*mồm|cái\s*miệng)\b',
    r'\bcâm\s*(mồm|miệng|mõm)\b',
]

# Tier 3: Moderate insults / derogatory terms
TIER3_MODERATE = [
    r'\bngu\s*(lắm|quá|vl|vcl|vãi|như\s*bò|như\s*chó|như\s*lợn|thế)\b',
    r'\bđần\s*(độn)?\b', r'\bngớ\s*ngẩn\b',
    r'\bhèn\s*(hạ|nhát|mọn)\b',
    r'\bnhục\s*(nhã)?\b',
    r'\bkinh\s*tởm\b', r'\bghê\s*tởm\b',
    r'\bxấu\s*xa\b', r'\bthối\s*nát\b',
    r'\bphản\s*động\b',
    r'\bbán\s*nước\b(?!.*phim)',  # "bán nước" but not in movie context
    r'\btay\s*sai\b',
    r'\bđê\s*tiện\b', r'\bti[eể]n\s*nhân\b',
    r'\bbẩn\s*thỉu\b', r'\bghê\b(?=.*\b(chết|quá)\b)',
]

# Context words that suggest the comment is about a movie/character, NOT personal attack
MOVIE_CONTEXT_WORDS = [
    'phim', 'nhân vật', 'cảnh', 'diễn', 'vai', 'đạo diễn', 'kịch bản',
    'review', 'trailer', 'rạp', 'chiếu', 'xem phim', 'bộ phim',
    'lịch sử', 'chiến tranh', 'thành cổ', 'quảng trị', 'mưa đỏ',
    'tiểu thuyết', 'truyện', 'tác phẩm', 'sản xuất',
    'sen', 'cường', 'tạ', 'quang', 'hồng', 'tú', 'tấn',  # character names
    'anh tạ', 'ông tạ', 'nhân vật sen', 'vai sen',
    'trong phim', 'trong truyện', 'đoạn phim', 'cảnh phim',
    'diễn xuất', 'thể loại', 'nội dung phim',
    'chiến sĩ', 'bộ đội', 'người lính', 'đồng đội', 'liệt sĩ',
]

# Words indicating the person is discussing movie plot, not attacking someone
NARRATIVE_INDICATORS = [
    'trong phim', 'trong truyện', 'nhân vật', 'vai diễn', 'cảnh này',
    'đoạn này', 'khúc này', 'phân cảnh', 'diễn viên', 'đóng vai',
    'hóa điên', 'hoá điên', 'giả điên', 'giả khùng', 'bị điên',
    'lịch sử', 'thực tế', 'chiến trường',
]

# Personal attack indicators - suggest comment targets a real person/commenter
PERSONAL_ATTACK_INDICATORS = [
    r'\bmày\b', r'\btao\b', r'\bchúng\s*mày\b',
    r'\bnó\s+là\b', r'\bbạn\s+là\b.*\bngu\b',
    r'\bông\s*à\b', r'\bbà\s*à\b',
    r'\bloại\s*(như\s*)?(mày|bạn|ông|bà)\b',
    r'\bngười\s*như\s*(mày|bạn)\b',
]

# "đéo" and "vcl" - can be used as emphasis (non-toxic) or insult (toxic)
# Need context analysis
CONTEXT_DEPENDENT = {
    'đéo': {
        'toxic_contexts': [
            r'đéo\s*(biết|hiểu).*gì',  # used dismissively AT someone
            r'mày.*đéo', r'đéo.*mày',   # combined with "mày"
        ],
        'nontoxic_contexts': [
            r'đéo\s*(sợ|ngán)',          # "not afraid" - bravado
            r't\s*đéo\b',               # casual self-expression
            r'đéo\s*(bao giờ|có)',       # emphasis "never"
        ]
    },
    'vcl': {
        'toxic_contexts': [
            r'ngu\s*vcl', r'vcl\s*ngu',
            r'ghét.*vcl', r'ác.*vcl',
        ],
        'nontoxic_contexts': [
            r'hay\s*vcl', r'vcl\s*hay',
            r'đẹp\s*vcl', r'ngon\s*vcl',
            r'cảm\s*động\s*vcl', r'khóc.*vcl',
            r'buồn\s*vcl',  # sad emphasis
        ]
    }
}

# ============================================================================
# CLASSIFIER
# ============================================================================

class ToxicClassifier:
    """Multi-layer Vietnamese toxic comment classifier."""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.tier1_patterns = [re.compile(p, re.IGNORECASE) for p in TIER1_EXPLICIT]
        self.tier2_patterns = [re.compile(p, re.IGNORECASE) for p in TIER2_STRONG_INSULTS]
        self.tier3_patterns = [re.compile(p, re.IGNORECASE) for p in TIER3_MODERATE]
        self.personal_patterns = [re.compile(p, re.IGNORECASE) for p in PERSONAL_ATTACK_INDICATORS]
        
        self.movie_context_lower = [w.lower() for w in MOVIE_CONTEXT_WORDS]
        self.narrative_lower = [w.lower() for w in NARRATIVE_INDICATORS]
        
        # Context dependent patterns
        self.context_dep = {}
        for word, contexts in CONTEXT_DEPENDENT.items():
            self.context_dep[word] = {
                'toxic': [re.compile(p, re.IGNORECASE) for p in contexts['toxic_contexts']],
                'nontoxic': [re.compile(p, re.IGNORECASE) for p in contexts['nontoxic_contexts']],
            }
    
    def _has_movie_context(self, text_lower: str) -> bool:
        """Check if text is discussing a movie/character."""
        count = sum(1 for w in self.movie_context_lower if w in text_lower)
        return count >= 2  # need at least 2 movie-context words
    
    def _has_narrative_context(self, text_lower: str) -> bool:
        """Check if text contains narrative/plot discussion indicators."""
        return any(w in text_lower for w in self.narrative_lower)
    
    def _has_personal_attack(self, text: str) -> bool:
        """Check if comment targets a specific person/commenter."""
        return any(p.search(text) for p in self.personal_patterns)
    
    def _check_quoting(self, text_lower: str) -> bool:
        """Check if toxic words appear to be quoted/referenced, not directly used."""
        quote_indicators = ['trích', 'nói rằng', 'câu nói', 'thoại', 'lời thoại',
                           'câu thoại', 'sợ cái', 'câu này']
        return any(q in text_lower for q in quote_indicators)
    
    def _check_tier1(self, text: str) -> list:
        """Check for Tier 1 (explicit profanity)."""
        matches = []
        for p in self.tier1_patterns:
            if p.search(text):
                matches.append(('tier1', p.pattern))
        return matches
    
    def _check_tier2(self, text: str) -> list:
        """Check for Tier 2 (strong insults)."""
        matches = []
        for p in self.tier2_patterns:
            if p.search(text):
                matches.append(('tier2', p.pattern))
        return matches
    
    def _check_tier3(self, text: str) -> list:
        """Check for Tier 3 (moderate insults)."""
        matches = []
        for p in self.tier3_patterns:
            if p.search(text):
                matches.append(('tier3', p.pattern))
        return matches
    
    def _check_context_dependent(self, text: str) -> tuple:
        """Analyze context-dependent words (đéo, vcl, etc.)."""
        toxic_score = 0
        nontoxic_score = 0
        signals = []
        
        text_lower = text.lower()
        
        for word, patterns in self.context_dep.items():
            if word in text_lower:
                for p in patterns['toxic']:
                    if p.search(text_lower):
                        toxic_score += 1
                        signals.append(('ctx_toxic', word))
                for p in patterns['nontoxic']:
                    if p.search(text_lower):
                        nontoxic_score += 1
                        signals.append(('ctx_nontoxic', word))
        
        return toxic_score, nontoxic_score, signals
    
    def _check_aggressive_punctuation(self, text: str) -> float:
        """Check for aggressive punctuation patterns."""
        score = 0.0
        # Many exclamation marks
        if text.count('!') >= 5:
            score += 0.1
        # ALL CAPS sections (Vietnamese)
        caps_words = re.findall(r'\b[A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴ]{2,}\b', text)
        if len(caps_words) >= 3:
            score += 0.1
        # Angry emoji
        angry_emoji = ['🤬', '😡', '💢', '🖕']
        if any(e in text for e in angry_emoji):
            score += 0.15
        return min(score, 0.3)
    
    def classify(self, text: str) -> LabelResult:
        """
        Classify a single comment as Toxic or Non-toxic.
        
        Uses multi-layer analysis:
        1. Check for explicit profanity (Tier 1)
        2. Check for strong insults (Tier 2)  
        3. Check for moderate insults (Tier 3)
        4. Analyze context (movie discussion vs personal attack)
        5. Handle context-dependent words
        6. Score and decide
        """
        result = LabelResult()
        
        if not text or not isinstance(text, str) or text.strip() == '':
            result.label = None
            result.label_name = ""
            result.confidence = 0.0
            result.needs_review = True
            result.reason = "Bình luận trống hoặc không hợp lệ"
            return result
        
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Very short comments (only emoji, single word, etc.)
        if len(text_stripped) <= 3 and not re.search(r'[a-zA-ZÀ-ỹ]', text_stripped):
            result.label = 0
            result.label_name = "Non-toxic"
            result.confidence = 0.7
            result.needs_review = False
            result.reason = "Bình luận quá ngắn, chỉ chứa emoji/ký tự đặc biệt"
            return result
        
        # ---- Layer 1: Explicit profanity ----
        tier1_matches = self._check_tier1(text_stripped)
        tier2_matches = self._check_tier2(text_stripped)
        tier3_matches = self._check_tier3(text_stripped)
        
        all_matches = tier1_matches + tier2_matches + tier3_matches
        
        # ---- Layer 2: Context analysis ----
        is_movie_context = self._has_movie_context(text_lower)
        is_narrative = self._has_narrative_context(text_lower)
        is_personal_attack = self._has_personal_attack(text_stripped)
        is_quoting = self._check_quoting(text_lower)
        
        # ---- Layer 3: Context-dependent words ----
        ctx_toxic, ctx_nontoxic, ctx_signals = self._check_context_dependent(text_stripped)
        
        # ---- Layer 4: Aggressive tone ----
        aggr_score = self._check_aggressive_punctuation(text_stripped)
        
        # ---- DECISION LOGIC ----
        
        toxic_score = 0.0
        reasons = []
        
        # Tier 1 matches: very strong signal
        if tier1_matches:
            toxic_score += 0.7
            result.toxic_signals.extend([m[1] for m in tier1_matches])
            reasons.append("Chứa từ ngữ tục tĩu/xúc phạm nghiêm trọng")
        
        # Tier 2 matches: strong signal
        if tier2_matches:
            toxic_score += 0.5
            result.toxic_signals.extend([m[1] for m in tier2_matches])
            reasons.append("Chứa lăng mạ/đe dọa trực tiếp")
        
        # Tier 3 matches: moderate signal
        if tier3_matches:
            toxic_score += 0.3
            result.toxic_signals.extend([m[1] for m in tier3_matches])
            reasons.append("Chứa từ ngữ xúc phạm mức trung bình")
        
        # Personal attack amplifier
        if is_personal_attack and all_matches:
            toxic_score += 0.2
            reasons.append("Nhắm vào cá nhân cụ thể (mày/tao)")
        
        # Context modifiers (reduce score if movie discussion)
        if is_movie_context and not is_personal_attack:
            toxic_score *= 0.4  # significantly reduce
            result.nontoxic_signals.append("Ngữ cảnh thảo luận phim")
            
        if is_narrative and not is_personal_attack:
            toxic_score *= 0.5
            result.nontoxic_signals.append("Ngữ cảnh kể chuyện/phân tích")
        
        if is_quoting:
            toxic_score *= 0.3
            result.nontoxic_signals.append("Trích dẫn lời thoại/câu nói")
        
        # Context-dependent words
        if ctx_toxic > ctx_nontoxic:
            toxic_score += 0.15
        elif ctx_nontoxic > ctx_toxic and toxic_score > 0:
            toxic_score *= 0.7
            result.nontoxic_signals.append("Từ nhấn mạnh dùng theo nghĩa tích cực")
        
        # Aggressive tone
        toxic_score += aggr_score
        
        # ---- FINAL CLASSIFICATION ----
        
        if toxic_score >= 0.5:
            result.label = 1
            result.label_name = "Toxic"
            result.confidence = min(0.95, 0.5 + toxic_score * 0.3)
            result.reason = "; ".join(reasons) if reasons else "Phát hiện dấu hiệu độc hại"
            
            # Flag for review if confidence is moderate
            if result.confidence < 0.7:
                result.needs_review = True
                result.reason += " [Cần kiểm tra lại - độ tin cậy trung bình]"
            
            # Flag if movie context conflicts with toxic signals
            if is_movie_context and tier1_matches:
                result.needs_review = True
                result.reason += " [Có ngữ cảnh phim nhưng chứa từ mạnh]"
                
        elif toxic_score >= 0.25:
            # Borderline - needs review
            result.label = 1
            result.label_name = "Toxic"
            result.confidence = 0.4 + toxic_score * 0.2
            result.needs_review = True
            result.reason = "; ".join(reasons) + " [Trường hợp biên - cần kiểm tra]" if reasons else "Dấu hiệu không rõ ràng"
            
        else:
            result.label = 0
            result.label_name = "Non-toxic"
            
            if all_matches and (is_movie_context or is_narrative or is_quoting):
                # Had matches but context overrode them
                result.confidence = 0.6
                result.needs_review = True
                result.reason = "Có từ nhạy cảm nhưng trong ngữ cảnh thảo luận phim/truyện"
            elif not all_matches:
                result.confidence = 0.85
                result.needs_review = False
                result.reason = "Không phát hiện dấu hiệu ngôn ngữ độc hại"
            else:
                result.confidence = 0.65
                result.needs_review = True
                result.reason = "Phát hiện từ nhạy cảm nhưng không đủ bằng chứng độc hại"
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_dataset(input_path: str, output_dir: str):
    """
    Process the entire dataset and generate labeled output files.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read data
    logger.info("Reading dataset from %s", input_path)
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    
    # Identify columns
    comment_col = 'comment'
    id_col = 'id_comment'
    
    if comment_col not in df.columns:
        raise ValueError(f"Column '{comment_col}' not found. Available: {df.columns.tolist()}")
    
    # Initialize classifier
    classifier = ToxicClassifier()
    
    # Process each comment
    logger.info("Classifying %d comments...", len(df))
    
    results = []
    for idx, row in df.iterrows():
        text = str(row[comment_col]) if pd.notna(row[comment_col]) else ""
        result = classifier.classify(text)
        results.append(result)
        
        if (idx + 1) % 5000 == 0:
            logger.info("Processed %d/%d comments", idx + 1, len(df))
    
    logger.info("Classification complete.")
    
    # Build output dataframe
    df_out = df.copy()
    
    # If there was an existing label column with data, rename it
    if 'label' in df_out.columns and df_out['label'].notna().any():
        df_out = df_out.rename(columns={'label': 'label_original'})
    elif 'label' in df_out.columns:
        df_out = df_out.drop(columns=['label'])
    
    df_out['label'] = [r.label for r in results]
    df_out['label_name'] = [r.label_name for r in results]
    df_out['confidence'] = [round(r.confidence, 3) for r in results]
    df_out['needs_review'] = [r.needs_review for r in results]
    df_out['reason'] = [r.reason for r in results]
    
    # Save full labeled dataset
    labeled_path = output_dir / 'facebook_comments_labeled.csv'
    df_out.to_csv(labeled_path, index=False, encoding='utf-8-sig')
    logger.info("Saved labeled dataset to %s", labeled_path)
    
    # Save review subset
    df_review = df_out[df_out['needs_review'] == True].copy()
    review_path = output_dir / 'facebook_comments_review.csv'
    df_review.to_csv(review_path, index=False, encoding='utf-8-sig')
    logger.info("Saved review dataset (%d comments) to %s", len(df_review), review_path)
    
    # Generate statistics
    total = len(df_out)
    toxic_count = (df_out['label'] == 1).sum()
    nontoxic_count = (df_out['label'] == 0).sum()
    undetermined = df_out['label'].isna().sum()
    review_count = df_out['needs_review'].sum()
    
    stats = {
        'total': int(total),
        'toxic': int(toxic_count),
        'nontoxic': int(nontoxic_count),
        'undetermined': int(undetermined),
        'review_needed': int(review_count),
        'toxic_pct': round(toxic_count / total * 100, 2) if total > 0 else 0,
        'nontoxic_pct': round(nontoxic_count / total * 100, 2) if total > 0 else 0,
        'avg_confidence': round(df_out['confidence'].mean(), 3),
    }
    
    logger.info("=== LABELING STATISTICS ===")
    logger.info("Total comments: %d", stats['total'])
    logger.info("Toxic: %d (%.2f%%)", stats['toxic'], stats['toxic_pct'])
    logger.info("Non-toxic: %d (%.2f%%)", stats['nontoxic'], stats['nontoxic_pct'])
    logger.info("Undetermined: %d", stats['undetermined'])
    logger.info("Needs review: %d", stats['review_needed'])
    logger.info("Average confidence: %.3f", stats['avg_confidence'])
    
    # Generate report
    generate_report(output_dir, stats, df_out)
    
    return df_out, stats


def generate_report(output_dir: Path, stats: dict, df: pd.DataFrame):
    """Generate the labeling report in markdown format."""
    
    # Sample toxic comments for report
    toxic_samples = df[df['label'] == 1].head(10)
    nontoxic_samples = df[df['label'] == 0].sample(min(5, len(df[df['label'] == 0])), random_state=42)
    review_samples = df[df['needs_review'] == True].head(10)
    
    # Confidence distribution
    conf_bins = pd.cut(df['confidence'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
    conf_dist = conf_bins.value_counts().sort_index()
    
    report = f"""# Báo Cáo Gán Nhãn Toxic / Non-toxic
## Dataset Bình Luận Facebook - Phim Mưa Đỏ

**Ngày thực hiện:** 2026-11-09  
**Người thực hiện:** Hệ thống tự động (AI-assisted rule-based classifier)  
**Dataset gốc:** `mua_do_comments_dedup.csv`

---

## 1. Tổng Quan Dataset

| Thông tin | Giá trị |
|---|---|
| Tổng số bình luận | {stats['total']:,} |
| Cột nội dung | `comment` |
| Cột ID | `id_comment` |
| Bình luận trống | 0 |
| Bình luận trùng lặp | 0 |
| Encoding gốc | UTF-8 with BOM |

## 2. Kết Quả Gán Nhãn

| Nhãn | Số lượng | Tỷ lệ |
|---|---|---|
| **Non-toxic (0)** | {stats['nontoxic']:,} | {stats['nontoxic_pct']}% |
| **Toxic (1)** | {stats['toxic']:,} | {stats['toxic_pct']}% |
| Chưa xác định | {stats['undetermined']} | - |
| **Cần kiểm tra lại** | {stats['review_needed']:,} | {round(stats['review_needed']/stats['total']*100, 2)}% |

**Độ tin cậy trung bình:** {stats['avg_confidence']}

### Phân bố độ tin cậy

| Khoảng | Số lượng |
|---|---|
"""
    for interval, count in conf_dist.items():
        report += f"| {interval} | {count:,} |\n"
    
    report += f"""
## 3. Tiêu Chí Gán Nhãn

### Toxic (label = 1)
- Chứa từ ngữ tục tĩu, xúc phạm nhằm tấn công cá nhân hoặc nhóm người.
- Lăng mạ, hạ nhục, sử dụng từ ngữ thô tục để công kích.
- Đe dọa, đe dọa bạo lực.
- Miệt thị, phân biệt đối xử, kích động thù ghét.
- Quấy rối, bắt nạt.

### Non-toxic (label = 0)
- Bình luận thông thường, trung lập.
- Khen ngợi, góp ý, trao đổi lịch sự.
- Phản biện, tranh luận không công kích cá nhân.
- Nhận xét về phim, sản phẩm mà không xúc phạm ai.
- Bình luận có cảm xúc tiêu cực nhưng không nhằm xúc phạm.

### Xử lý đặc biệt
- Từ ngữ trong ngữ cảnh thảo luận phim (ví dụ: "nhân vật hóa điên") → **Non-toxic**.
- Trích dẫn lời thoại phim → **Non-toxic**.
- Từ nhấn mạnh tiếng lóng ("vcl", "đéo") dùng để bày tỏ cảm xúc tích cực → xét ngữ cảnh.
- Phản biện gay gắt nhưng không công kích cá nhân → **Non-toxic**.

## 4. Phương Pháp Phân Loại

### Kiến trúc hệ thống
Hệ thống phân loại đa tầng (multi-layer classification):

1. **Tầng 1 - Từ vựng tục tĩu rõ ràng (Tier 1):** Phát hiện các từ ngữ xúc phạm nghiêm trọng.
2. **Tầng 2 - Lăng mạ/đe dọa mạnh (Tier 2):** Phát hiện các cấu trúc lăng mạ trực tiếp.
3. **Tầng 3 - Xúc phạm mức trung bình (Tier 3):** Phát hiện từ ngữ xúc phạm ở mức vừa phải.
4. **Phân tích ngữ cảnh:** Xem xét ngữ cảnh thảo luận phim, trích dẫn, kể chuyện.
5. **Phát hiện tấn công cá nhân:** Nhận diện các chỉ báo nhắm vào cá nhân cụ thể (mày/tao).
6. **Từ phụ thuộc ngữ cảnh:** Phân tích các từ như "đéo", "vcl" theo ngữ cảnh sử dụng.
7. **Tính điểm và quyết định:** Kết hợp tất cả tín hiệu để đưa ra nhãn cuối cùng.

### Công cụ sử dụng
- Python 3 với pandas, regex
- Hệ thống rule-based đa tầng với phân tích ngữ cảnh
- **KHÔNG** phải keyword matching đơn thuần

### Quan trọng
- Nhãn do hệ thống AI heuristic tạo ra, **CHƯA** được con người xác nhận.
- Các bình luận có `needs_review = TRUE` cần được kiểm tra thủ công.
- Hệ thống ưu tiên recall cho Toxic (sẵn sàng gán nhãn Toxic hơn là bỏ sót).

## 5. Mẫu Bình Luận Toxic

| ID | Bình luận (rút gọn) | Confidence | Lý do |
|---|---|---|---|
"""
    for _, row in toxic_samples.iterrows():
        cmt = str(row['comment'])[:80].replace('|', '\\|').replace('\n', ' ')
        report += f"| {row['id_comment']} | {cmt}... | {row['confidence']} | {row['reason'][:60]} |\n"
    
    report += f"""
## 6. Mẫu Bình Luận Cần Kiểm Tra

| ID | Bình luận (rút gọn) | Label | Confidence | Lý do |
|---|---|---|---|---|
"""
    for _, row in review_samples.iterrows():
        cmt = str(row['comment'])[:70].replace('|', '\\|').replace('\n', ' ')
        report += f"| {row['id_comment']} | {cmt}... | {row['label_name']} | {row['confidence']} | {row['reason'][:50]} |\n"
    
    report += f"""
## 7. Hạn Chế và Rủi Ro

1. **Không có ground truth:** Nhãn được tạo bởi hệ thống heuristic, chưa được con người xác nhận.
2. **Giới hạn ngữ cảnh:** Hệ thống không hiểu được mỉa mai, chế giễu phức tạp, hoặc ngữ cảnh ngoài văn bản.
3. **Teen code chưa đầy đủ:** Một số biến thể viết tắt, viết lệch để né kiểm duyệt có thể bị bỏ sót.
4. **Thiên lệch dataset:** Dataset chủ yếu là bình luận về phim Mưa Đỏ, nhiều từ nhạy cảm được dùng trong ngữ cảnh phim.
5. **False positive:** Một số bình luận non-toxic có thể bị gán nhãn toxic do chứa từ nhạy cảm.
6. **False negative:** Một số bình luận toxic tinh vi (mỉa mai, công kích gián tiếp) có thể bị bỏ sót.

## 8. Khuyến Nghị

1. **Kiểm tra thủ công** tất cả bình luận có `needs_review = TRUE` ({stats['review_needed']:,} bình luận).
2. **Ưu tiên kiểm tra** các bình luận toxic có `confidence < 0.7`.
3. **Rà soát ngẫu nhiên** khoảng 5-10% bình luận non-toxic để phát hiện false negative.
4. Sau khi kiểm tra xong, cập nhật file `facebook_comments_labeled.csv` với nhãn đã xác nhận.

## 9. File Đầu Ra

| File | Mô tả |
|---|---|
| `facebook_comments_labeled.csv` | Toàn bộ dataset với nhãn |
| `facebook_comments_review.csv` | Các bình luận cần kiểm tra lại |
| `labeling_report.md` | Báo cáo này |
"""
    
    report_path = output_dir / 'labeling_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info("Saved report to %s", report_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    input_file = Path(r'd:\btl kpdl\mua_do_comments_dedup.csv')
    output_directory = Path(r'd:\btl kpdl\labeling')
    
    df_labeled, statistics = process_dataset(str(input_file), str(output_directory))
    
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Total: {statistics['total']:,}")
    print(f"Toxic: {statistics['toxic']:,} ({statistics['toxic_pct']}%)")
    print(f"Non-toxic: {statistics['nontoxic']:,} ({statistics['nontoxic_pct']}%)")
    print(f"Needs review: {statistics['review_needed']:,}")
    print(f"Avg confidence: {statistics['avg_confidence']}")
    print(f"\nOutput files in: {output_directory}")

