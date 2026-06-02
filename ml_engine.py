"""
ml_engine.py

Advanced Machine Learning Engine for Expense Categorization
Implements ensemble methods, local training, and sophisticated prediction strategies.

Features:
- Ensemble predictions combining multiple ML models
- Local model training with user data
- Feature engineering for better categorization
- Confidence scoring and explainability
- Continuous learning from user corrections
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class EnsembleCategorizationEngine:
    """
    Ensemble-based categorization engine that combines multiple prediction methods
    for improved accuracy and robustness.
    """
    
    def __init__(self):
        self.rule_based_model = RuleBasedClassifier()
        self.pattern_model = PatternMatchingClassifier()
        self.similarity_model = SimilarityClassifier()
        self.frequency_model = FrequencyBasedClassifier()
        
        # Weights for ensemble (can be learned over time)
        self.model_weights = {
            'rule_based': 0.30,
            'pattern': 0.25,
            'similarity': 0.25,
            'frequency': 0.20
        }
        
        # Performance tracking
        self.model_performance = {
            'rule_based': {'correct': 0, 'total': 0},
            'pattern': {'correct': 0, 'total': 0},
            'similarity': {'correct': 0, 'total': 0},
            'frequency': {'correct': 0, 'total': 0}
        }
    
    def predict(
        self, 
        transaction: Dict, 
        historical_data: Optional[List[Dict]] = None
    ) -> Tuple[str, str, float, Dict]:
        """
        Predict category using ensemble of models.
        
        Returns:
            (category, subcategory, confidence, explanation)
        """
        predictions = {}
        
        # Get predictions from each model
        predictions['rule_based'] = self.rule_based_model.predict(transaction)
        predictions['pattern'] = self.pattern_model.predict(transaction)
        predictions['similarity'] = self.similarity_model.predict(transaction, historical_data)
        predictions['frequency'] = self.frequency_model.predict(transaction, historical_data)
        
        # Combine predictions using weighted voting
        category_scores = defaultdict(float)
        subcategory_scores = defaultdict(lambda: defaultdict(float))
        
        for model_name, (cat, subcat, conf) in predictions.items():
            weight = self.model_weights[model_name]
            weighted_conf = conf * weight
            
            category_scores[cat] += weighted_conf
            if subcat:
                subcategory_scores[cat][subcat] += weighted_conf
        
        # Get best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        category = best_category[0]
        category_confidence = best_category[1]
        
        # Get best subcategory for the chosen category
        subcategory = None
        if category in subcategory_scores and subcategory_scores[category]:
            best_subcat = max(subcategory_scores[category].items(), key=lambda x: x[1])
            subcategory = best_subcat[0]
        
        # Normalize confidence to 0-1 range
        total_weight = sum(self.model_weights.values())
        final_confidence = min(1.0, category_confidence / total_weight)
        
        # Build explanation
        explanation = self._build_explanation(predictions, category, subcategory, final_confidence)
        
        return category, subcategory, final_confidence, explanation
    
    @staticmethod
    def _category_from_prediction(pred) -> str:
        """Extract category from raw model tuple or explanation dict entry."""
        if isinstance(pred, dict):
            return pred.get("category", "Uncategorised")
        cat, _, _ = pred
        return cat

    def learn_from_correction(
        self, 
        transaction: Dict, 
        predicted_category: str, 
        actual_category: str,
        model_predictions: Dict
    ):
        """Update models based on user corrections."""
        # Update performance metrics
        for model_name, pred in model_predictions.items():
            cat = self._category_from_prediction(pred)
            self.model_performance[model_name]['total'] += 1
            if cat == actual_category:
                self.model_performance[model_name]['correct'] += 1
        
        # Update individual model weights based on performance
        self._update_weights()
        
        # Train individual models
        self.rule_based_model.learn(transaction, actual_category)
        self.pattern_model.learn(transaction, actual_category)
        self.similarity_model.learn(transaction, actual_category)
        self.frequency_model.learn(transaction, actual_category)
    
    def _update_weights(self):
        """Dynamically adjust model weights based on performance."""
        total_accuracy = 0
        accuracies = {}
        
        for model_name, perf in self.model_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                accuracies[model_name] = accuracy
                total_accuracy += accuracy
        
        if total_accuracy > 0:
            # Redistribute weights based on relative accuracy
            for model_name in self.model_weights:
                if model_name in accuracies:
                    self.model_weights[model_name] = accuracies[model_name] / total_accuracy
    
    def _build_explanation(
        self, 
        predictions: Dict, 
        final_category: str, 
        final_subcategory: Optional[str],
        confidence: float
    ) -> Dict:
        """Build human-readable explanation of prediction."""
        explanation = {
            'final_category': final_category,
            'final_subcategory': final_subcategory,
            'confidence': confidence,
            'model_predictions': {},
            'agreement_level': 0,
            'reasoning': []
        }
        
        # Count how many models agreed on the final category
        agreement_count = sum(1 for _, (cat, _, _) in predictions.items() if cat == final_category)
        explanation['agreement_level'] = agreement_count / len(predictions)
        
        # Add individual model predictions
        for model_name, (cat, subcat, conf) in predictions.items():
            explanation['model_predictions'][model_name] = {
                'category': cat,
                'subcategory': subcat,
                'confidence': conf,
                'agreed': cat == final_category
            }
            
            # Add reasoning for models that agreed
            if cat == final_category:
                explanation['reasoning'].append(
                    f"{model_name.replace('_', ' ').title()}: {cat} ({conf:.1%} confidence)"
                )
        
        return explanation
    
    def get_model_stats(self) -> Dict:
        """Get performance statistics for all models."""
        stats = {}
        for model_name, perf in self.model_performance.items():
            if perf['total'] > 0:
                stats[model_name] = {
                    'accuracy': perf['correct'] / perf['total'],
                    'total_predictions': perf['total'],
                    'correct_predictions': perf['correct'],
                    'current_weight': self.model_weights[model_name]
                }
        return stats


class RuleBasedClassifier:
    """Rule-based classification using keyword matching."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.custom_rules = []
    
    def predict(self, transaction: Dict) -> Tuple[str, Optional[str], float]:
        """Predict using rule-based matching."""
        description = transaction.get('description', '').lower()
        amount = abs(float(transaction.get('amount', 0)))
        
        # Check custom rules first (higher priority)
        for rule in self.custom_rules:
            if self._match_rule(description, amount, rule):
                return rule['category'], rule.get('subcategory'), rule['confidence']
        
        # Check predefined rules
        for rule in self.rules:
            if self._match_rule(description, amount, rule):
                return rule['category'], rule.get('subcategory'), 0.8
        
        return "Uncategorised", None, 0.0
    
    def learn(self, transaction: Dict, category: str):
        """Learn new rules from corrections."""
        description = transaction.get('description', '').lower()
        
        # Extract key terms
        terms = self._extract_key_terms(description)
        
        if terms:
            # Create new custom rule
            self.custom_rules.append({
                'keywords': terms,
                'category': category,
                'subcategory': transaction.get('subcategory'),
                'confidence': 0.9,
                'examples': 1
            })
    
    def _match_rule(self, description: str, amount: float, rule: Dict) -> bool:
        """Check if transaction matches a rule."""
        # Check keywords
        if 'keywords' in rule:
            if not any(keyword in description for keyword in rule['keywords']):
                return False
        
        # Check amount range if specified
        if 'amount_min' in rule and amount < rule['amount_min']:
            return False
        if 'amount_max' in rule and amount > rule['amount_max']:
            return False
        
        return True
    
    def _extract_key_terms(self, description: str) -> List[str]:
        """Extract meaningful terms from description (supports English and Japanese)."""
        # Extract English words (3+ letters)
        en_words = re.findall(r'\b[a-z]{3,}\b', description)
        # Extract Japanese character runs (katakana / hiragana / kanji, 2+ chars)
        jp_words = re.findall(r'[\u3040-\u30FF\u4E00-\u9FFF]{2,}', description)

        stop_words = {'the', 'and', 'for', 'with', 'from', 'card', 'use', 'visa', 'domestic'}
        key_terms = [w for w in en_words if w not in stop_words] + jp_words

        return key_terms[:4]

    def _initialize_rules(self) -> List[Dict]:
        """Initialize default categorization rules aligned with the 10 app categories.

        Categories: Food, Social Life, Subscriptions, Household, Transportation,
                    Vacation, Health, Apparel, Grooming, Self-development
        """
        return [
            # ── Food ──
            {'keywords': ['ローソン', 'lawson'], 'category': 'Food', 'subcategory': 'Groceries'},
            {'keywords': ['セブンイレブン', '7-eleven'], 'category': 'Food', 'subcategory': 'Groceries'},
            {'keywords': ['ファミリーマート', 'familymart'], 'category': 'Food', 'subcategory': 'Groceries'},
            {'keywords': ['スーパー', 'supermarket', 'grocery', 'イオン', 'aeon', 'イトーヨーカドー',
                          '西友', 'seiyu', 'ライフ', 'マルエツ', 'サミット', 'オーケー', 'コストコ',
                          '業務スーパー'],
             'category': 'Food', 'subcategory': 'Groceries'},
            {'keywords': ['レストラン', 'restaurant', '居酒屋', 'izakaya', 'dinner'],
             'category': 'Food', 'subcategory': 'Dinner/Eating Out'},
            {'keywords': ['lunch', 'ランチ', '昼食', 'カフェ', 'cafe'],
             'category': 'Food', 'subcategory': 'Lunch/Eating Out'},
            {'keywords': ['スターバックス', 'starbucks', 'スタバ', 'タリーズ', 'tully', 'ドトール',
                          'doutor', 'coffee', 'コーヒー'],
             'category': 'Food', 'subcategory': 'Beverages A'},
            {'keywords': ['マクドナルド', "mcdonald", 'モスバーガー', 'mos burger', 'すき家',
                          'sukiya', '吉野家', 'yoshinoya', '松屋', 'matsuya', 'サイゼリヤ',
                          'saizeriya', 'ガスト', 'gusto', 'ココイチ', 'coco'],
             'category': 'Food', 'subcategory': 'Dinner/Eating Out'},
            {'keywords': ['コンビニ', 'ミニストップ', 'ministop', 'デイリーヤマザキ', 'ポプラ'],
             'category': 'Food', 'subcategory': 'Groceries'},

            # ── Social Life ──
            {'keywords': ['飲み会', 'drinking', 'パーティー', 'party', 'カラオケ', 'karaoke',
                          'ボーリング', 'bowling'],
             'category': 'Social Life', 'subcategory': 'Drinking'},
            {'keywords': ['イベント', 'event', '会食', 'dining', '懇親会', 'networking',
                          '歓迎会', 'welcome', '送別会', 'farewell'],
             'category': 'Social Life', 'subcategory': 'Event'},

            # ── Subscriptions ──
            {'keywords': ['netflix', 'spotify', 'hulu', 'disney', 'youtube premium',
                          'amazon prime', 'アマゾンプライム', 'icloud', 'apple music',
                          'google one', 'subscription', 'membership', '月額', '年額',
                          '年会費', 'annual fee'],
             'category': 'Subscriptions', 'subcategory': 'Digital Services'},
            {'keywords': ['楽天カード', '楽天ゴールドカード', 'rakuten card', 'credit card fee'],
             'category': 'Subscriptions', 'subcategory': 'Credit Card'},

            # ── Household ──
            {'keywords': ['家賃', 'rent', '住宅', 'housing'],
             'category': 'Household', 'subcategory': 'Rent'},
            {'keywords': ['光熱費', 'utility', '電気', 'electric', 'tepco', '東京電力',
                          'ガス', 'tokyo gas', '東京ガス', '水道', 'water', '東京都水道'],
             'category': 'Household', 'subcategory': 'Utilities'},
            {'keywords': ['ソフトバンク', 'softbank', 'ドコモ', 'docomo', 'エーユー',
                          '楽天モバイル', 'rakuten mobile', 'phone', 'mobile', 'internet',
                          'broadband'],
             'category': 'Household', 'subcategory': 'Utilities'},
            {'keywords': ['家具', 'furniture', 'ニトリ', 'nitori', 'イケア', 'ikea',
                          'ホームセンター', 'home center'],
             'category': 'Household', 'subcategory': 'Furniture'},
            {'keywords': ['日用品', 'daily necessities', 'ダイソー', 'daiso', 'キャンドゥ',
                          'セリア', '100 yen', 'cleaning'],
             'category': 'Household', 'subcategory': 'Daily Necessities'},

            # ── Transportation ──
            {'keywords': ['電車', 'train', '地下鉄', 'subway', 'モノレール', 'monorail',
                          'suica', 'スイカ', 'pasmo', 'パスモ', 'モバイルsuica', 'mobile suica'],
             'category': 'Transportation', 'subcategory': 'Subway'},
            {'keywords': ['タクシー', 'taxi', 'uber', 'grab', 'ライドシェア'],
             'category': 'Transportation', 'subcategory': 'Taxi'},
            {'keywords': ['ＥＴＣ', 'etc', '高速道路', 'highway', '駐車場', 'parking',
                          'ガソリン', 'gasoline', '燃料', 'fuel', 'petrol'],
             'category': 'Transportation', 'subcategory': 'ETC'},
            {'keywords': ['バス', 'bus', '交通費', 'transport', 'モバイルパス', 'mobile pass'],
             'category': 'Transportation', 'subcategory': 'Mobile Pass'},

            # ── Vacation ──
            {'keywords': ['旅行', 'travel', 'ホテル', 'hotel', '飛行機', 'flight',
                          '新幹線', 'shinkansen', '観光', 'tourism', '温泉', 'onsen',
                          'リゾート', 'resort', 'チケット', 'ticket', 'ツアー', 'tour',
                          '宿泊', 'accommodation', 'airbnb', 'booking.com'],
             'category': 'Vacation', 'subcategory': 'Travel'},

            # ── Health ──
            {'keywords': ['病院', 'hospital', 'クリニック', 'clinic', '歯科', 'dental',
                          'dentist', '眼科', 'eye', '薬局', 'pharmacy', '薬', 'medicine',
                          '診察', 'examination', '治療', 'treatment', 'medical', 'doctor'],
             'category': 'Health', 'subcategory': 'Medical'},
            {'keywords': ['フィットネス', 'fitness', 'ジム', 'gym', 'anytime fitness',
                          'ヨガ', 'yoga', 'マッサージ', 'massage'],
             'category': 'Health', 'subcategory': 'Fitness'},

            # ── Apparel ──
            {'keywords': ['服', 'clothing', '靴', 'shoes', 'バッグ', 'bag', 'アクセサリー',
                          'accessory', '時計', 'watch', 'ユニクロ', 'uniqlo', 'ジーユー',
                          'しまむら', 'shimamura', 'zara', 'h&m', 'gap', 'nike', 'adidas',
                          'ナイキ', 'アディダス', 'ファッション', 'fashion', 'zozotown'],
             'category': 'Apparel', 'subcategory': 'Clothing'},

            # ── Grooming ──
            {'keywords': ['美容', 'beauty', '化粧品', 'cosmetics', 'スキンケア', 'skincare',
                          'ネイル', 'nail', 'エステ', 'esthetic', '理容', 'barber',
                          '美容院', 'salon', '資生堂', 'shiseido', 'マツモトキヨシ',
                          'matsumoto kiyoshi', 'ウエルシア', 'welcia', 'ツルハ', 'サンドラッグ',
                          'ココカラファイン'],
             'category': 'Grooming', 'subcategory': 'Personal Care'},

            # ── Self-development ──
            {'keywords': ['本', 'book', '雑誌', 'magazine', '新聞', 'newspaper',
                          '講座', 'course', 'セミナー', 'seminar', 'ワークショップ', 'workshop',
                          '資格', 'certification', '学習', 'learning', 'スキル', 'skill',
                          'トレーニング', 'training', 'elearning', 'cinema', 'movie',
                          '映画', 'theater'],
             'category': 'Self-development', 'subcategory': 'Learning'},

            # ── Online shopping (general) ──
            {'keywords': ['amazon', 'アマゾン', '楽天', 'rakuten', 'ヤフー', 'yahoo',
                          'メルカリ', 'mercari'],
             'category': 'Food', 'subcategory': 'Groceries'},  # default; often re-categorized by user
        ]


class PatternMatchingClassifier:
    """Pattern-based classification using regex and text patterns."""
    
    def __init__(self):
        self.patterns = {}  # category -> list of regex patterns
        self.learned_patterns = defaultdict(list)
    
    def predict(self, transaction: Dict) -> Tuple[str, Optional[str], float]:
        """Predict using pattern matching."""
        description = transaction.get('description', '')
        
        # Check learned patterns first
        for category, patterns in self.learned_patterns.items():
            for pattern, confidence in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    return category, None, confidence
        
        return "Uncategorised", None, 0.0
    
    def learn(self, transaction: Dict, category: str):
        """Learn new patterns from transactions."""
        description = transaction.get('description', '').lower()
        
        # Extract potential patterns
        # Look for repeated structures
        words = description.split()
        if len(words) >= 2:
            # Create pattern from first few words
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in words[:2]) + r'\b'
            self.learned_patterns[category].append((pattern, 0.75))


class SimilarityClassifier:
    """Classification based on similarity to known transactions."""
    
    def __init__(self):
        self.known_transactions = []
    
    def predict(
        self, 
        transaction: Dict, 
        historical_data: Optional[List[Dict]] = None
    ) -> Tuple[str, Optional[str], float]:
        """Predict based on similarity to historical transactions."""
        if not historical_data:
            return "Uncategorised", None, 0.0
        
        description = transaction.get('description', '').lower()
        amount = abs(float(transaction.get('amount', 0)))
        
        best_match = None
        best_similarity = 0.0
        
        for hist_trans in historical_data:
            if not hist_trans.get('category'):
                continue
            
            # Calculate similarity
            hist_desc = hist_trans.get('description', '').lower()
            desc_similarity = self._calculate_similarity(description, hist_desc)
            
            # Consider amount similarity
            hist_amount = abs(float(hist_trans.get('amount', 0)))
            amount_similarity = 1.0 - min(1.0, abs(amount - hist_amount) / max(amount, hist_amount, 1))
            
            # Combined similarity (weighted)
            total_similarity = (desc_similarity * 0.7) + (amount_similarity * 0.3)
            
            if total_similarity > best_similarity:
                best_similarity = total_similarity
                best_match = hist_trans
        
        if best_match and best_similarity > 0.6:
            return (
                best_match['category'], 
                best_match.get('subcategory'),
                best_similarity
            )
        
        return "Uncategorised", None, 0.0
    
    def learn(self, transaction: Dict, category: str):
        """Store transaction for future similarity matching."""
        self.known_transactions.append({
            'description': transaction.get('description'),
            'amount': transaction.get('amount'),
            'category': category,
            'subcategory': transaction.get('subcategory')
        })
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using token-based approach."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()


class FrequencyBasedClassifier:
    """Classification based on frequency patterns of merchants and amounts."""
    
    def __init__(self):
        self.merchant_frequency = defaultdict(lambda: defaultdict(int))
        self.amount_patterns = defaultdict(list)
    
    def predict(
        self, 
        transaction: Dict, 
        historical_data: Optional[List[Dict]] = None
    ) -> Tuple[str, Optional[str], float]:
        """Predict based on frequency patterns."""
        if not historical_data:
            return "Uncategorised", None, 0.0
        
        # Build frequency model from historical data
        self._update_frequency_model(historical_data)
        
        description = transaction.get('description', '').lower()
        merchant = self._extract_merchant(description)
        
        if merchant in self.merchant_frequency:
            # Get most frequent category for this merchant
            category_counts = self.merchant_frequency[merchant]
            if category_counts:
                most_frequent = max(category_counts.items(), key=lambda x: x[1])
                category = most_frequent[0]
                total_count = sum(category_counts.values())
                confidence = most_frequent[1] / total_count
                
                return category, None, confidence
        
        return "Uncategorised", None, 0.0
    
    def learn(self, transaction: Dict, category: str):
        """Update frequency statistics."""
        description = transaction.get('description', '').lower()
        merchant = self._extract_merchant(description)
        
        self.merchant_frequency[merchant][category] += 1
    
    def _update_frequency_model(self, historical_data: List[Dict]):
        """Update frequency model from historical data."""
        for trans in historical_data:
            if trans.get('category'):
                description = trans.get('description', '').lower()
                merchant = self._extract_merchant(description)
                self.merchant_frequency[merchant][trans['category']] += 1
    
    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name from description."""
        # Remove common prefixes
        desc = re.sub(r'(visa domestic use vs|credit card|debit card|atm|pos)\s+', '', description)
        # Get first significant word
        words = desc.split()
        return words[0] if words else "unknown"


class LocalMLTrainer:
    """
    Local machine learning trainer for privacy-preserving model updates.
    Trains models on user's device without sending data to external services.
    """
    
    def __init__(self):
        self.training_data = []
        self.feature_names = []
        self.label_encoder = {}
        self.model = None
    
    def add_training_example(self, transaction: Dict, category: str, subcategory: Optional[str] = None):
        """Add a training example."""
        self.training_data.append({
            'transaction': transaction,
            'category': category,
            'subcategory': subcategory
        })
    
    def extract_features(self, transaction: Dict) -> Dict:
        """Extract numerical features from transaction for ML."""
        features = {}
        
        # Amount-based features
        amount = abs(float(transaction.get('amount', 0)))
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        features['amount_rounded'] = int(amount / 100) * 100
        
        # Description-based features
        description = transaction.get('description', '').lower()
        features['desc_length'] = len(description)
        features['desc_word_count'] = len(description.split())
        features['has_numbers'] = int(bool(re.search(r'\d', description)))
        
        # Time-based features (if date available)
        if 'date' in transaction:
            try:
                date = pd.to_datetime(transaction['date'])
                features['day_of_week'] = date.dayofweek
                features['day_of_month'] = date.day
                features['month'] = date.month
                features['is_weekend'] = int(date.dayofweek >= 5)
            except:
                pass
        
        return features
    
    def train(self):
        """Train local model on accumulated data."""
        if len(self.training_data) < 10:
            return False  # Need minimum data to train
        
        # For now, this is a placeholder for actual ML training
        # In production, this would use sklearn or similar
        return True
    
    def predict(self, transaction: Dict) -> Tuple[str, float]:
        """Predict using locally trained model."""
        # Placeholder for ML prediction
        return "Uncategorised", 0.0
    
    def get_training_stats(self) -> Dict:
        """Get statistics about training data."""
        if not self.training_data:
            return {
                'total_examples': 0,
                'categories': [],
                'ready_to_train': False
            }
        
        categories = Counter([ex['category'] for ex in self.training_data])
        
        return {
            'total_examples': len(self.training_data),
            'categories': dict(categories),
            'ready_to_train': len(self.training_data) >= 10,
            'most_common_category': categories.most_common(1)[0] if categories else None
        }

