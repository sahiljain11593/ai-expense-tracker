#!/usr/bin/env python3
"""
Smart Learning System for Transaction Categorization
Combines Machine Learning with Rule-Based Learning
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

class SmartLearningSystem:
    """Advanced learning system for transaction categorization."""
    
    def __init__(self, model_path: str = "learning_models/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        # Learning databases
        self.merchant_patterns = {}
        self.user_corrections = []
        self.historical_patterns = {}
        
        # ML Models
        self.text_classifier = None
        self.amount_classifier = None
        self.ensemble_weights = {'text': 0.4, 'rules': 0.3, 'patterns': 0.3}
        
        # Load existing models if available
        self.load_models()
    
    def learn_from_user_feedback(self, transaction_id: str, original_prediction: str, 
                                user_correction: str, transaction_data: dict) -> None:
        """Learn from user corrections to improve future predictions."""
        
        correction = {
            'transaction_id': transaction_id,
            'original_prediction': original_prediction,
            'user_correction': user_correction,
            'transaction_data': transaction_data,
            'timestamp': datetime.now().isoformat(),
            'confidence': self.calculate_confidence(transaction_data)
        }
        
        self.user_corrections.append(correction)
        
        # Update merchant patterns
        merchant = transaction_data.get('description', '').lower()
        if merchant not in self.merchant_patterns:
            self.merchant_patterns[merchant] = {
                'category': user_correction,
                'confidence': 1.0,
                'count': 1,
                'last_used': datetime.now().isoformat()
            }
        else:
            # Update existing pattern
            pattern = self.merchant_patterns[merchant]
            pattern['count'] += 1
            pattern['last_used'] = datetime.now().isoformat()
            
            # Increase confidence with more confirmations
            if pattern['category'] == user_correction:
                pattern['confidence'] = min(1.0, pattern['confidence'] + 0.1)
            else:
                # Category changed, reset confidence
                pattern['confidence'] = 0.5
                pattern['category'] = user_correction
        
        # Save updated patterns
        self.save_merchant_patterns()
    
    def predict_category(self, transaction_data: dict) -> Tuple[str, float, Dict]:
        """Predict category using ensemble of ML and rule-based methods."""
        
        # Get predictions from different methods
        text_prediction = self.predict_with_text_ml(transaction_data)
        rule_prediction = self.predict_with_rules(transaction_data)
        pattern_prediction = self.predict_with_patterns(transaction_data)
        
        # Ensemble voting
        predictions = [text_prediction, rule_prediction, pattern_prediction]
        categories = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        # Weighted ensemble
        final_category = self.ensemble_vote(categories, confidences)
        final_confidence = self.calculate_ensemble_confidence(categories, confidences)
        
        # Detailed breakdown
        breakdown = {
            'text_ml': {'category': text_prediction[0], 'confidence': text_prediction[1]},
            'rules': {'category': rule_prediction[0], 'confidence': rule_prediction[1]},
            'patterns': {'category': pattern_prediction[0], 'confidence': pattern_prediction[1]},
            'ensemble': {'category': final_category, 'confidence': final_confidence}
        }
        
        return final_category, final_confidence, breakdown
    
    def predict_with_text_ml(self, transaction_data: dict) -> Tuple[str, float]:
        """Predict category using text-based machine learning."""
        
        if self.text_classifier is None:
            return "Uncategorised", 0.0
        
        try:
            # Prepare text features
            description = transaction_data.get('description', '')
            original_desc = transaction_data.get('original_description', '')
            text = f"{description} {original_desc}".strip()
            
            if not text:
                return "Uncategorised", 0.0
            
            # Vectorize text
            text_vector = self.text_vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.text_classifier.predict(text_vector)[0]
            probabilities = self.text_classifier.predict_proba(text_vector)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Text ML prediction error: {e}")
            return "Uncategorised", 0.0
    
    def predict_with_rules(self, transaction_data: dict) -> Tuple[str, float]:
        """Predict category using rule-based system."""
        
        description = str(transaction_data.get('description', '')).lower()
        original_desc = str(transaction_data.get('original_description', '')).lower()
        
        # Check merchant patterns first (highest confidence)
        merchant = description.lower()
        if merchant in self.merchant_patterns:
            pattern = self.merchant_patterns[merchant]
            return pattern['category'], pattern['confidence']
        
        # Check keyword rules
        rules = self.get_categorization_rules()
        for category, keywords in rules.items():
            if any(keyword.lower() in description or keyword.lower() in original_desc 
                   for keyword in keywords):
                return category, 0.7  # Medium confidence for keyword matches
        
        return "Uncategorised", 0.0
    
    def predict_with_patterns(self, transaction_data: dict) -> Tuple[str, float]:
        """Predict category using historical patterns."""
        
        # Time-based patterns
        date = transaction_data.get('date')
        if date:
            weekday = date.weekday() if hasattr(date, 'weekday') else 0
            time_pattern = self.get_time_based_patterns(weekday)
            if time_pattern:
                return time_pattern['category'], time_pattern['confidence']
        
        # Amount-based patterns
        amount = transaction_data.get('amount', 0)
        amount_pattern = self.get_amount_based_patterns(amount)
        if amount_pattern:
            return amount_pattern['category'], amount_pattern['confidence']
        
        return "Uncategorised", 0.0
    
    def ensemble_vote(self, categories: List[str], confidences: List[float]) -> str:
        """Combine predictions from different methods using weighted voting."""
        
        if not categories:
            return "Uncategorised"
        
        # Weighted voting
        category_scores = {}
        for i, (category, confidence) in enumerate(zip(categories, confidences)):
            weight = list(self.ensemble_weights.values())[i]
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += confidence * weight
        
        # Return category with highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def calculate_ensemble_confidence(self, categories: List[str], confidences: List[float]) -> float:
        """Calculate overall confidence for ensemble prediction."""
        
        if not confidences:
            return 0.0
        
        # Weighted average confidence
        weighted_confidences = []
        for i, confidence in enumerate(confidences):
            weight = list(self.ensemble_weights.values())[i]
            weighted_confidences.append(confidence * weight)
        
        return sum(weighted_confidences) / sum(self.ensemble_weights.values())
    
    def train_models(self, training_data: pd.DataFrame) -> None:
        """Train machine learning models with user data."""
        
        if training_data.empty:
            return
        
        try:
            # Prepare text features
            descriptions = training_data['description'].fillna('') + ' ' + \
                         training_data.get('original_description', '').fillna('')
            
            # Text vectorization
            self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            text_features = self.text_vectorizer.fit_transform(descriptions)
            
            # Train text classifier
            self.text_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.text_classifier.fit(text_features, training_data['category'])
            
            # Save models
            self.save_models()
            
        except Exception as e:
            print(f"Model training error: {e}")
    
    def get_categorization_rules(self) -> Dict[str, List[str]]:
        """Get current categorization rules."""
        
        return {
            "Food": [
                "ローソン", "セブンイレブン", "ファミリーマート", "コンビニ", "lawson", 
                "seven eleven", "family mart", "ポプラグループ", "poplar", "スーパー", 
                "supermarket", "grocery", "market", "food", "fresh", "イオン", "aeon"
            ],
            "Transportation": [
                "電車", "train", "バス", "bus", "タクシー", "taxi", "地下鉄", "subway",
                "モバイルパス", "mobile pass", "交通費", "transport", "ＥＴＣ", "etc"
            ],
            "Subscriptions": [
                "icloud", "apple music", "amazon prime", "google one", "netflix", 
                "spotify", "subscription", "membership", "月額", "monthly"
            ],
            "Household": [
                "家賃", "rent", "光熱費", "utility", "電気", "electric", "ガス", "gas",
                "家具", "furniture", "ニトリ", "nitori", "イケア", "ikea"
            ]
        }
    
    def get_time_based_patterns(self, weekday: int) -> Optional[Dict]:
        """Get time-based spending patterns."""
        
        # Example patterns (would be learned from data)
        patterns = {
            5: {'category': 'Food', 'confidence': 0.6},  # Saturday = groceries
            6: {'category': 'Food', 'confidence': 0.6},  # Sunday = groceries
            0: {'category': 'Transportation', 'confidence': 0.5}  # Monday = commute
        }
        
        return patterns.get(weekday)
    
    def get_amount_based_patterns(self, amount: float) -> Optional[Dict]:
        """Get amount-based spending patterns."""
        
        # Example patterns (would be learned from data)
        if amount < 1000:
            return {'category': 'Food', 'confidence': 0.5}
        elif amount < 5000:
            return {'category': 'Food', 'confidence': 0.4}
        elif amount > 50000:
            return {'category': 'Household', 'confidence': 0.6}
        
        return None
    
    def calculate_confidence(self, transaction_data: dict) -> float:
        """Calculate confidence score for a transaction."""
        
        # Base confidence
        confidence = 0.5
        
        # Boost confidence for known merchants
        merchant = transaction_data.get('description', '').lower()
        if merchant in self.merchant_patterns:
            confidence += 0.3
        
        # Boost confidence for clear descriptions
        description = transaction_data.get('description', '')
        if len(description) > 10:
            confidence += 0.1
        
        # Boost confidence for typical amounts
        amount = transaction_data.get('amount', 0)
        if 100 <= amount <= 10000:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        
        try:
            if self.text_classifier:
                with open(f"{self.model_path}/text_classifier.pkl", 'wb') as f:
                    pickle.dump(self.text_classifier, f)
            
            if self.text_vectorizer:
                with open(f"{self.model_path}/text_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.text_vectorizer, f)
                
        except Exception as e:
            print(f"Model save error: {e}")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        
        try:
            # Load text classifier
            classifier_path = f"{self.model_path}/text_classifier.pkl"
            if os.path.exists(classifier_path):
                with open(classifier_path, 'rb') as f:
                    self.text_classifier = pickle.load(f)
            
            # Load text vectorizer
            vectorizer_path = f"{self.model_path}/text_vectorizer.pkl"
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.text_vectorizer = pickle.load(f)
                    
        except Exception as e:
            print(f"Model load error: {e}")
    
    def save_merchant_patterns(self) -> None:
        """Save merchant patterns to disk."""
        
        try:
            with open(f"{self.model_path}/merchant_patterns.json", 'w') as f:
                json.dump(self.merchant_patterns, f, indent=2)
        except Exception as e:
            print(f"Pattern save error: {e}")
    
    def load_merchant_patterns(self) -> None:
        """Load merchant patterns from disk."""
        
        try:
            pattern_path = f"{self.model_path}/merchant_patterns.json"
            if os.path.exists(pattern_path):
                with open(pattern_path, 'r') as f:
                    self.merchant_patterns = json.load(f)
        except Exception as e:
            print(f"Pattern load error: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system."""
        
        return {
            'total_corrections': len(self.user_corrections),
            'merchant_patterns': len(self.merchant_patterns),
            'models_trained': self.text_classifier is not None,
            'last_training': self.get_last_training_date()
        }
    
    def get_last_training_date(self) -> str:
        """Get the last training date."""
        
        if not self.user_corrections:
            return "Never"
        
        # Find the latest correction
        latest = max(self.user_corrections, key=lambda x: x['timestamp'])
        return latest['timestamp'][:10]  # Just the date part
