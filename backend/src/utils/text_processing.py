import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import string
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Advanced text processing utilities for multilingual research papers
    """
    
    def __init__(self):
        # Common stop words in multiple languages
        self.stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'},
            'de': {'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'einer', 'eines', 'und', 'oder', 'aber', 'in', 'an', 'auf', 'zu', 'für', 'von', 'mit', 'bei', 'nach', 'über', 'unter', 'zwischen', 'ist', 'sind', 'war', 'waren', 'sein', 'haben', 'hat', 'hatte', 'hatten', 'werden', 'wird', 'wurde', 'wurden', 'dieser', 'diese', 'dieses'},
            'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'dans', 'sur', 'à', 'pour', 'avec', 'par', 'entre', 'est', 'sont', 'était', 'étaient', 'être', 'avoir', 'a', 'avait', 'avaient', 'sera', 'seront', 'ce', 'cette', 'ces'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'del', 'de', 'y', 'o', 'pero', 'en', 'sobre', 'a', 'para', 'con', 'por', 'entre', 'es', 'son', 'era', 'eran', 'ser', 'tener', 'tiene', 'tenía', 'tenían', 'será', 'serán', 'este', 'esta', 'estos', 'estas'},
            'zh': {'的', '和', '或', '但', '在', '上', '到', '为', '与', '由', '从', '之间', '是', '被', '有', '将', '会', '能', '可以', '这', '那', '这些', '那些'}
        }
        
        # Academic keywords that should be preserved
        self.academic_keywords = {
            'en': {'research', 'study', 'analysis', 'method', 'approach', 'model', 'algorithm', 'system', 'framework', 'technique', 'evaluation', 'experiment', 'result', 'conclusion', 'finding', 'application', 'development', 'implementation', 'performance', 'optimization', 'machine', 'learning', 'neural', 'network', 'deep', 'artificial', 'intelligence', 'data', 'mining', 'classification', 'regression', 'clustering', 'prediction', 'accuracy', 'precision', 'recall'},
            'de': {'forschung', 'studie', 'analyse', 'methode', 'ansatz', 'modell', 'algorithmus', 'system', 'rahmen', 'technik', 'bewertung', 'experiment', 'ergebnis', 'schlussfolgerung', 'befund', 'anwendung', 'entwicklung', 'implementierung', 'leistung', 'optimierung'},
            'fr': {'recherche', 'étude', 'analyse', 'méthode', 'approche', 'modèle', 'algorithme', 'système', 'cadre', 'technique', 'évaluation', 'expérience', 'résultat', 'conclusion', 'découverte', 'application', 'développement', 'implémentation', 'performance', 'optimisation'},
            'es': {'investigación', 'estudio', 'análisis', 'método', 'enfoque', 'modelo', 'algoritmo', 'sistema', 'marco', 'técnica', 'evaluación', 'experimento', 'resultado', 'conclusión', 'hallazgo', 'aplicación', 'desarrollo', 'implementación', 'rendimiento', 'optimización'},
            'zh': {'研究', '学习', '分析', '方法', '模型', '算法', '系统', '框架', '技术', '评估', '实验', '结果', '结论', '发现', '应用', '开发', '实现', '性能', '优化', '机器', '神经', '网络', '深度', '人工智能', '数据', '挖掘', '分类', '回归', '聚类', '预测', '准确性'}
        }
        
        logger.info("📝 Text processor initialized with multilingual support")
    
    def clean_text(self, text: str, preserve_academic: bool = True, 
                   language: str = 'en') -> str:
        """
        Clean and normalize text while preserving academic content
        
        Args:
            text: Input text to clean
            preserve_academic: Whether to preserve academic keywords
            language: Language code for language-specific processing
            
        Returns:
            Cleaned text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but preserve some punctuation
            if language == 'zh':
                # For Chinese, preserve Chinese characters and basic punctuation
                text = re.sub(r'[^\u4e00-\u9fff\w\s\.,;:!?()-]', '', text)
            else:
                # For other languages, preserve letters, numbers, and basic punctuation
                text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)
            
            # Remove URLs and email addresses
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[.]{3,}', '...', text)
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            # Clean up spacing around punctuation
            text = re.sub(r'\s+([,.;:!?])', r'\1', text)
            text = re.sub(r'([,.;:!?])\s*([,.;:!?])', r'\1 \2', text)
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Text cleaning failed: {e}")
            return str(text) if text else ""
    
    def extract_keywords(self, text: str, language: str = 'en', 
                        max_keywords: int = 20, 
                        min_word_length: int = 3) -> List[str]:
        """
        Extract important keywords from text
        
        Args:
            text: Input text
            language: Language code
            max_keywords: Maximum number of keywords to return
            min_word_length: Minimum word length to consider
            
        Returns:
            List of extracted keywords
        """
        try:
            if not text:
                return []
            
            # Clean text first
            cleaned_text = self.clean_text(text, language=language).lower()
            
            # Tokenize
            if language == 'zh':
                # For Chinese, split by characters and whitespace
                words = re.findall(r'[\u4e00-\u9fff]+|\w+', cleaned_text)
            else:
                # For other languages, split by whitespace and punctuation
                words = re.findall(r'\b\w+\b', cleaned_text)
            
            # Filter words
            stop_words = self.stop_words.get(language, self.stop_words['en'])
            academic_keywords = self.academic_keywords.get(language, self.academic_keywords['en'])
            
            filtered_words = []
            for word in words:
                word = word.lower().strip()
                if (len(word) >= min_word_length and 
                    word not in stop_words and 
                    not word.isdigit() and
                    re.match(r'^[a-zA-Z\u4e00-\u9fff]+$', word)):
                    filtered_words.append(word)
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Boost academic keywords
            for word in academic_keywords:
                if word in word_counts:
                    word_counts[word] *= 2  # Double the weight of academic keywords
            
            # Get top keywords
            top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
            
            logger.info(f"🔤 Extracted {len(top_keywords)} keywords from {language} text")
            return top_keywords
            
        except Exception as e:
            logger.error(f"❌ Keyword extraction failed: {e}")
            return []
    
    def calculate_text_similarity(self, text1: str, text2: str, 
                                 language: str = 'en') -> float:
        """
        Calculate similarity between two texts using keyword overlap
        
        Args:
            text1: First text
            text2: Second text
            language: Language code
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Extract keywords from both texts
            keywords1 = set(self.extract_keywords(text1, language=language, max_keywords=30))
            keywords2 = set(self.extract_keywords(text2, language=language, max_keywords=30))
            
            if not keywords1 or not keywords2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = keywords1.intersection(keywords2)
            union = keywords1.union(keywords2)
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"❌ Text similarity calculation failed: {e}")
            return 0.0
    
    def extract_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """
        Extract meaningful sentences from text
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences to return
            
        Returns:
            List of extracted sentences
        """
        try:
            if not text:
                return []
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', cleaned_text)
            
            # Filter and clean sentences
            meaningful_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence.split()) >= 5 and  # At least 5 words
                    len(sentence) >= 30 and         # At least 30 characters
                    not sentence.isupper()):       # Not all caps
                    meaningful_sentences.append(sentence)
            
            # Return top sentences by length (longer sentences tend to be more informative)
            meaningful_sentences.sort(key=len, reverse=True)
            
            return meaningful_sentences[:max_sentences]
            
        except Exception as e:
            logger.error(f"❌ Sentence extraction failed: {e}")
            return []
    
    def detect_research_domain_keywords(self, text: str, language: str = 'en') -> Dict[str, float]:
        """
        Detect research domain based on keyword analysis
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary mapping domains to confidence scores
        """
        try:
            # Domain-specific keyword mappings
            domain_keywords = {
                'Computer Vision': {
                    'en': ['image', 'vision', 'visual', 'cnn', 'object', 'detection', 'segmentation', 'classification', 'feature', 'pixel', 'convolutional', 'opencv', 'recognition'],
                    'zh': ['图像', '视觉', '视频', '识别', '检测', '分割', '特征', '像素']
                },
                'Natural Language Processing': {
                    'en': ['text', 'language', 'nlp', 'bert', 'transformer', 'embedding', 'sentiment', 'parsing', 'tokenization', 'word', 'sentence', 'linguistic', 'corpus'],
                    'zh': ['文本', '语言', '词汇', '句子', '语义', '语法', '分词']
                },
                'Machine Learning': {
                    'en': ['learning', 'model', 'algorithm', 'training', 'neural', 'optimization', 'prediction', 'supervised', 'unsupervised', 'regression', 'classification', 'clustering'],
                    'zh': ['学习', '模型', '算法', '训练', '预测', '监督', '无监督', '回归', '分类', '聚类']
                },
                'Deep Learning': {
                    'en': ['deep', 'neural', 'network', 'layers', 'backpropagation', 'gradient', 'activation', 'dropout', 'batch', 'epoch', 'tensorflow', 'pytorch'],
                    'zh': ['深度', '神经', '网络', '层', '梯度', '激活', '训练']
                },
                'Healthcare': {
                    'en': ['medical', 'health', 'clinical', 'patient', 'diagnosis', 'treatment', 'disease', 'symptom', 'therapy', 'pharmaceutical', 'epidemiology'],
                    'zh': ['医疗', '健康', '临床', '患者', '诊断', '治疗', '疾病', '症状']
                },
                'Robotics': {
                    'en': ['robot', 'robotic', 'autonomous', 'control', 'navigation', 'manipulation', 'sensor', 'actuator', 'kinematics', 'dynamics'],
                    'zh': ['机器人', '自主', '控制', '导航', '操作', '传感器']
                }
            }
            
            # Extract keywords from text
            text_keywords = set(self.extract_keywords(text, language=language, max_keywords=50))
            text_lower = text.lower()
            
            # Calculate domain scores
            domain_scores = {}
            for domain, lang_keywords in domain_keywords.items():
                keywords = lang_keywords.get(language, lang_keywords.get('en', []))
                
                # Count keyword matches
                matches = 0
                total_keywords = len(keywords)
                
                for keyword in keywords:
                    if keyword.lower() in text_keywords or keyword.lower() in text_lower:
                        matches += 1
                
                # Calculate confidence score
                confidence = matches / total_keywords if total_keywords > 0 else 0.0
                domain_scores[domain] = confidence
            
            # Normalize scores
            max_score = max(domain_scores.values()) if domain_scores.values() else 1.0
            if max_score > 0:
                domain_scores = {domain: score / max_score for domain, score in domain_scores.items()}
            
            return domain_scores
            
        except Exception as e:
            logger.error(f"❌ Domain detection failed: {e}")
            return {}
    
    def extract_acronyms_and_abbreviations(self, text: str) -> List[str]:
        """
        Extract acronyms and abbreviations from text
        
        Args:
            text: Input text
            
        Returns:
            List of found acronyms and abbreviations
        """
        try:
            if not text:
                return []
            
            # Pattern for acronyms (2-6 capital letters)
            acronym_pattern = r'\b[A-Z]{2,6}\b'
            acronyms = re.findall(acronym_pattern, text)
            
            # Pattern for abbreviations with periods
            abbrev_pattern = r'\b[A-Za-z]\.(?:[A-Za-z]\.)+[A-Za-z]?\b'
            abbreviations = re.findall(abbrev_pattern, text)
            
            # Combine and deduplicate
            all_abbrevs = list(set(acronyms + abbreviations))
            
            # Filter out common false positives
            false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAS', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'USE', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'HIM', 'TWO', 'HOW', 'ITS', 'WHO', 'DID', 'YES', 'HIS', 'HAS', 'HAD'}
            
            filtered_abbrevs = [abbrev for abbrev in all_abbrevs if abbrev.upper() not in false_positives]
            
            return filtered_abbrevs[:20]  # Limit to top 20
            
        except Exception as e:
            logger.error(f"❌ Acronym extraction failed: {e}")
            return []
    
    def summarize_text(self, text: str, max_sentences: int = 3, 
                      language: str = 'en') -> str:
        """
        Create a simple extractive summary of the text
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum sentences in summary
            language: Language code
            
        Returns:
            Summary text
        """
        try:
            if not text:
                return ""
            
            # Extract meaningful sentences
            sentences = self.extract_sentences(text, max_sentences * 2)
            
            if not sentences:
                return text[:200] + "..." if len(text) > 200 else text
            
            # Score sentences based on keyword density
            keywords = set(self.extract_keywords(text, language=language, max_keywords=20))
            
            sentence_scores = []
            for sentence in sentences:
                sentence_keywords = set(self.extract_keywords(sentence, language=language))
                overlap = len(sentence_keywords.intersection(keywords))
                score = overlap / len(sentence_keywords) if sentence_keywords else 0
                sentence_scores.append((sentence, score))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]
            
            # Join sentences
            summary = ' '.join(top_sentences)
            
            return summary if summary else text[:300] + "..." if len(text) > 300 else text
            
        except Exception as e:
            logger.error(f"❌ Text summarization failed: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def validate_academic_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """
        Validate if text appears to be academic/research content
        
        Args:
            text: Input text to validate
            language: Language code
            
        Returns:
            Validation results with confidence score
        """
        try:
            if not text:
                return {'is_academic': False, 'confidence': 0.0, 'reasons': ['Empty text']}
            
            reasons = []
            score = 0.0
            
            # Check length (academic texts are usually substantial)
            if len(text) > 500:
                score += 0.2
                reasons.append('Substantial length')
            
            # Check for academic keywords
            academic_keywords = self.academic_keywords.get(language, self.academic_keywords['en'])
            text_lower = text.lower()
            academic_keyword_count = sum(1 for keyword in academic_keywords if keyword in text_lower)
            
            if academic_keyword_count > 5:
                score += 0.3
                reasons.append(f'Contains {academic_keyword_count} academic keywords')
            elif academic_keyword_count > 2:
                score += 0.15
                reasons.append(f'Contains {academic_keyword_count} academic keywords')
            
            # Check for citations or references (basic patterns)
            citation_patterns = [
                r'\([12][0-9]{3}\)',  # Year in parentheses
                r'\bet al\.',         # Et al.
                r'\bref\.',           # Ref.
                r'\bfig\.',           # Fig.
                r'\btable\s+\d+',     # Table number
                r'\bequation\s+\d+',  # Equation number
            ]
            
            citation_count = 0
            for pattern in citation_patterns:
                citation_count += len(re.findall(pattern, text_lower))
            
            if citation_count > 3:
                score += 0.2
                reasons.append(f'Contains {citation_count} citation-like patterns')
            
            # Check for technical terminology
            technical_terms = ['algorithm', 'method', 'approach', 'framework', 'model', 'system', 'analysis', 'evaluation', 'experiment', 'result', 'conclusion']
            technical_count = sum(1 for term in technical_terms if term in text_lower)
            
            if technical_count > 3:
                score += 0.2
                reasons.append(f'Contains {technical_count} technical terms')
            
            # Check for formal structure indicators
            structure_indicators = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references']
            structure_count = sum(1 for indicator in structure_indicators if indicator in text_lower)
            
            if structure_count > 2:
                score += 0.1
                reasons.append(f'Contains {structure_count} structural indicators')
            
            # Determine if text is academic
            is_academic = score >= 0.5
            confidence = min(1.0, score)
            
            return {
                'is_academic': is_academic,
                'confidence': confidence,
                'score': score,
                'reasons': reasons,
                'academic_keyword_count': academic_keyword_count,
                'citation_count': citation_count,
                'technical_term_count': technical_count
            }
            
        except Exception as e:
            logger.error(f"❌ Academic text validation failed: {e}")
            return {'is_academic': False, 'confidence': 0.0, 'reasons': ['Validation error'], 'error': str(e)}
    
    def preprocess_for_embedding(self, text: str, language: str = 'en', 
                                max_length: int = 500) -> str:
        """
        Preprocess text for embedding generation
        
        Args:
            text: Input text
            language: Language code
            max_length: Maximum text length
            
        Returns:
            Preprocessed text ready for embedding
        """
        try:
            if not text:
                return ""
            
            # Clean text
            cleaned = self.clean_text(text, preserve_academic=True, language=language)
            
            # Extract key sentences if text is too long
            if len(cleaned) > max_length:
                sentences = self.extract_sentences(cleaned, max_sentences=5)
                if sentences:
                    cleaned = ' '.join(sentences)
            
            # Final truncation if still too long
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + "..."
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Embedding preprocessing failed: {e}")
            return text[:max_length] if text else ""

# Utility functions for backward compatibility
def clean_text(text: str, language: str = 'en') -> str:
    """Utility function for text cleaning"""
    processor = TextProcessor()
    return processor.clean_text(text, language=language)

def extract_keywords(text: str, language: str = 'en', max_keywords: int = 20) -> List[str]:
    """Utility function for keyword extraction"""
    processor = TextProcessor()
    return processor.extract_keywords(text, language=language, max_keywords=max_keywords)

def calculate_similarity(text1: str, text2: str, language: str = 'en') -> float:
    """Utility function for text similarity calculation"""
    processor = TextProcessor()
    return processor.calculate_text_similarity(text1, text2, language=language)
