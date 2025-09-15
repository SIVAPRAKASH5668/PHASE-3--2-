import logging
from typing import Dict, List, Any, Optional, Tuple
import pycountry
from collections import defaultdict

logger = logging.getLogger(__name__)

class LanguageUtils:
    """
    Comprehensive language utilities for multilingual research processing
    """
    
    def __init__(self):
        # Extended language mappings with regional variants
        self.language_mappings = {
            # Primary languages with full support
            'en': {'name': 'English', 'family': 'Germanic', 'script': 'Latin', 'priority': 1.0, 'regions': ['US', 'UK', 'CA', 'AU']},
            'zh': {'name': 'Chinese', 'family': 'Sino-Tibetan', 'script': 'Han', 'priority': 0.95, 'regions': ['CN', 'TW', 'HK', 'SG']},
            'de': {'name': 'German', 'family': 'Germanic', 'script': 'Latin', 'priority': 0.9, 'regions': ['DE', 'AT', 'CH']},
            'fr': {'name': 'French', 'family': 'Romance', 'script': 'Latin', 'priority': 0.9, 'regions': ['FR', 'CA', 'BE', 'CH']},
            'ja': {'name': 'Japanese', 'family': 'Japonic', 'script': 'Kanji/Hiragana/Katakana', 'priority': 0.85, 'regions': ['JP']},
            'ko': {'name': 'Korean', 'family': 'Koreanic', 'script': 'Hangul', 'priority': 0.8, 'regions': ['KR']},
            'es': {'name': 'Spanish', 'family': 'Romance', 'script': 'Latin', 'priority': 0.85, 'regions': ['ES', 'MX', 'AR', 'CO']},
            'ru': {'name': 'Russian', 'family': 'Slavic', 'script': 'Cyrillic', 'priority': 0.8, 'regions': ['RU']},
            'it': {'name': 'Italian', 'family': 'Romance', 'script': 'Latin', 'priority': 0.75, 'regions': ['IT']},
            'pt': {'name': 'Portuguese', 'family': 'Romance', 'script': 'Latin', 'priority': 0.75, 'regions': ['BR', 'PT']},
            'ar': {'name': 'Arabic', 'family': 'Semitic', 'script': 'Arabic', 'priority': 0.7, 'regions': ['SA', 'AE', 'EG']},
            'hi': {'name': 'Hindi', 'family': 'Indo-European', 'script': 'Devanagari', 'priority': 0.7, 'regions': ['IN']},
            
            # Secondary languages with partial support
            'nl': {'name': 'Dutch', 'family': 'Germanic', 'script': 'Latin', 'priority': 0.65, 'regions': ['NL', 'BE']},
            'sv': {'name': 'Swedish', 'family': 'Germanic', 'script': 'Latin', 'priority': 0.6, 'regions': ['SE']},
            'no': {'name': 'Norwegian', 'family': 'Germanic', 'script': 'Latin', 'priority': 0.6, 'regions': ['NO']},
            'da': {'name': 'Danish', 'family': 'Germanic', 'script': 'Latin', 'priority': 0.6, 'regions': ['DK']},
            'fi': {'name': 'Finnish', 'family': 'Finno-Ugric', 'script': 'Latin', 'priority': 0.55, 'regions': ['FI']},
            'tr': {'name': 'Turkish', 'family': 'Turkic', 'script': 'Latin', 'priority': 0.55, 'regions': ['TR']},
            'pl': {'name': 'Polish', 'family': 'Slavic', 'script': 'Latin', 'priority': 0.55, 'regions': ['PL']},
            'cs': {'name': 'Czech', 'family': 'Slavic', 'script': 'Latin', 'priority': 0.5, 'regions': ['CZ']},
            'hu': {'name': 'Hungarian', 'family': 'Finno-Ugric', 'script': 'Latin', 'priority': 0.5, 'regions': ['HU']},
        }
        
        # Research domain language preferences
        self.domain_language_preferences = {
            'Computer Science': ['en', 'zh', 'de', 'ja', 'ko'],
            'Medicine': ['en', 'de', 'fr', 'zh', 'ja'],
            'Engineering': ['en', 'de', 'zh', 'ja', 'ko'],
            'Physics': ['en', 'de', 'fr', 'ru', 'zh'],
            'Biology': ['en', 'de', 'fr', 'zh', 'ja'],
            'Mathematics': ['en', 'de', 'fr', 'ru', 'zh'],
            'Economics': ['en', 'de', 'fr', 'es', 'zh'],
            'Psychology': ['en', 'de', 'fr', 'es', 'it'],
        }
        
        # Language similarity matrix for related language processing
        self.language_similarity = {
            'en': {'de': 0.6, 'nl': 0.7, 'sv': 0.5, 'no': 0.5, 'da': 0.5},
            'de': {'en': 0.6, 'nl': 0.8, 'sv': 0.6, 'no': 0.6, 'da': 0.6},
            'fr': {'es': 0.8, 'it': 0.8, 'pt': 0.8, 'ro': 0.7},
            'es': {'fr': 0.8, 'it': 0.8, 'pt': 0.9, 'ca': 0.9},
            'it': {'fr': 0.8, 'es': 0.8, 'pt': 0.7, 'ro': 0.7},
            'pt': {'es': 0.9, 'fr': 0.8, 'it': 0.7},
            'ru': {'uk': 0.9, 'be': 0.8, 'pl': 0.6, 'cs': 0.6, 'sk': 0.7},
            'zh': {'ja': 0.3, 'ko': 0.2},  # Lower similarity due to different language families
        }
        
        logger.info("🌍 Language utilities initialized with comprehensive language support")
    
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a language
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            Dictionary with language information
        """
        try:
            if language_code in self.language_mappings:
                base_info = self.language_mappings[language_code].copy()
                
                # Add additional information from pycountry if available
                try:
                    lang_obj = pycountry.languages.get(alpha_2=language_code)
                    if lang_obj:
                        base_info['iso_name'] = lang_obj.name
                        base_info['alpha_3'] = getattr(lang_obj, 'alpha_3', '')
                except:
                    pass
                
                base_info['code'] = language_code
                return base_info
            else:
                # Return basic info for unknown languages
                return {
                    'code': language_code,
                    'name': f'Language ({language_code})',
                    'family': 'Unknown',
                    'script': 'Unknown',
                    'priority': 0.3,
                    'regions': []
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get language info for {language_code}: {e}")
            return {'code': language_code, 'name': 'Unknown', 'priority': 0.3}
    
    def get_research_priority_languages(self, domain: str = None) -> List[Dict[str, Any]]:
        """
        Get languages prioritized for research in a specific domain
        
        Args:
            domain: Research domain (optional)
            
        Returns:
            List of prioritized languages with metadata
        """
        try:
            if domain and domain in self.domain_language_preferences:
                priority_codes = self.domain_language_preferences[domain]
            else:
                # Default priority based on general research significance
                priority_codes = ['en', 'zh', 'de', 'fr', 'ja', 'ko', 'es', 'ru', 'it', 'pt', 'ar', 'hi']
            
            priority_languages = []
            for code in priority_codes:
                lang_info = self.get_language_info(code)
                if domain:
                    # Boost priority for domain-specific preferences
                    lang_info['domain_priority'] = len(priority_codes) - priority_codes.index(code)
                priority_languages.append(lang_info)
            
            return priority_languages
            
        except Exception as e:
            logger.error(f"❌ Failed to get research priority languages: {e}")
            return [self.get_language_info('en')]  # Fallback to English
    
    def suggest_related_languages(self, base_language: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest related languages for expanded search
        
        Args:
            base_language: Base language code
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of related language codes
        """
        try:
            suggestions = []
            
            # Check direct similarity mappings
            if base_language in self.language_similarity:
                similar_langs = self.language_similarity[base_language]
                # Sort by similarity score
                sorted_langs = sorted(similar_langs.items(), key=lambda x: x[1], reverse=True)
                suggestions.extend([lang for lang, score in sorted_langs[:max_suggestions]])
            
            # Add languages from the same family
            base_info = self.get_language_info(base_language)
            base_family = base_info.get('family', '')
            
            if base_family and base_family != 'Unknown':
                family_languages = [
                    code for code, info in self.language_mappings.items()
                    if info.get('family') == base_family and code != base_language
                ]
                
                # Add family languages not already in suggestions
                for lang in family_languages:
                    if lang not in suggestions:
                        suggestions.append(lang)
                        if len(suggestions) >= max_suggestions:
                            break
            
            # Fill remaining slots with high-priority languages
            if len(suggestions) < max_suggestions:
                high_priority = ['en', 'zh', 'de', 'fr', 'ja', 'es']
                for lang in high_priority:
                    if lang not in suggestions and lang != base_language:
                        suggestions.append(lang)
                        if len(suggestions) >= max_suggestions:
                            break
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"❌ Failed to suggest related languages for {base_language}: {e}")
            return ['en'] if base_language != 'en' else ['zh', 'de', 'fr']
    
    def calculate_language_coverage_score(self, languages: List[str], domain: str = None) -> float:
        """
        Calculate coverage score for a set of languages in research context
        
        Args:
            languages: List of language codes
            domain: Research domain (optional)
            
        Returns:
            Coverage score between 0 and 1
        """
        try:
            if not languages:
                return 0.0
            
            # Get priority languages for domain
            priority_languages = self.get_research_priority_languages(domain)
            priority_codes = [lang['code'] for lang in priority_languages]
            
            # Calculate coverage
            covered_priorities = sum(1 for lang in languages if lang in priority_codes[:10])  # Top 10
            total_priorities = min(10, len(priority_codes))
            
            # Base coverage score
            base_coverage = covered_priorities / total_priorities if total_priorities > 0 else 0.0
            
            # Bonus for language diversity
            families = set()
            scripts = set()
            
            for lang in languages:
                info = self.get_language_info(lang)
                families.add(info.get('family', 'Unknown'))
                scripts.add(info.get('script', 'Unknown'))
            
            # Diversity bonus (up to 0.2 additional points)
            family_diversity = min(0.1, len(families) * 0.02)
            script_diversity = min(0.1, len(scripts) * 0.02)
            
            final_score = min(1.0, base_coverage + family_diversity + script_diversity)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate language coverage score: {e}")
            return 0.5  # Default coverage
    
    def group_languages_by_family(self, languages: List[str]) -> Dict[str, List[str]]:
        """
        Group languages by their language family
        
        Args:
            languages: List of language codes
            
        Returns:
            Dictionary mapping families to language lists
        """
        try:
            family_groups = defaultdict(list)
            
            for lang in languages:
                info = self.get_language_info(lang)
                family = info.get('family', 'Unknown')
                family_groups[family].append(lang)
            
            return dict(family_groups)
            
        except Exception as e:
            logger.error(f"❌ Failed to group languages by family: {e}")
            return {'Unknown': languages}
    
    def get_translation_difficulty(self, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Estimate translation difficulty between two languages
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with difficulty assessment
        """
        try:
            source_info = self.get_language_info(source_lang)
            target_info = self.get_language_info(target_lang)
            
            # Base difficulty factors
            difficulty_score = 0.5  # Base difficulty
            
            # Same language family = easier
            if source_info.get('family') == target_info.get('family'):
                difficulty_score -= 0.2
            
            # Same script = easier
            if source_info.get('script') == target_info.get('script'):
                difficulty_score -= 0.1
            
            # Check direct similarity
            if source_lang in self.language_similarity:
                if target_lang in self.language_similarity[source_lang]:
                    similarity = self.language_similarity[source_lang][target_lang]
                    difficulty_score -= similarity * 0.3
            
            # Special cases
            if source_lang == target_lang:
                difficulty_score = 0.0  # No translation needed
            elif source_lang == 'en' or target_lang == 'en':
                difficulty_score -= 0.1  # English is well-supported
            
            # Clamp score
            difficulty_score = max(0.0, min(1.0, difficulty_score))
            
            # Classify difficulty
            if difficulty_score < 0.3:
                difficulty_class = "Easy"
            elif difficulty_score < 0.6:
                difficulty_class = "Moderate"
            else:
                difficulty_class = "Difficult"
            
            return {
                'difficulty_score': difficulty_score,
                'difficulty_class': difficulty_class,
                'source_family': source_info.get('family', 'Unknown'),
                'target_family': target_info.get('family', 'Unknown'),
                'source_script': source_info.get('script', 'Unknown'),
                'target_script': target_info.get('script', 'Unknown'),
                'similarity_score': self.language_similarity.get(source_lang, {}).get(target_lang, 0.0),
                'recommendations': self._get_translation_recommendations(difficulty_score, source_lang, target_lang)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to assess translation difficulty: {e}")
            return {
                'difficulty_score': 0.5,
                'difficulty_class': 'Moderate',
                'error': str(e)
            }
    
    def _get_translation_recommendations(self, difficulty_score: float, 
                                       source_lang: str, target_lang: str) -> List[str]:
        """Generate recommendations for translation"""
        recommendations = []
        
        if difficulty_score < 0.3:
            recommendations.append("Translation should be straightforward")
            recommendations.append("Standard translation services should work well")
        elif difficulty_score < 0.6:
            recommendations.append("Consider using multiple translation services for comparison")
            recommendations.append("Review translations for context accuracy")
        else:
            recommendations.append("Use specialized translation services for these languages")
            recommendations.append("Consider human review for critical translations")
            recommendations.append("Break complex sentences into simpler parts")
        
        return recommendations
    
    def optimize_language_selection(self, available_languages: List[str], 
                                  target_coverage: float = 0.8,
                                  domain: str = None) -> List[str]:
        """
        Optimize language selection for maximum research coverage
        
        Args:
            available_languages: Available language options
            target_coverage: Target coverage score (0-1)
            domain: Research domain for optimization
            
        Returns:
            Optimized list of languages
        """
        try:
            if not available_languages:
                return ['en']  # Fallback to English
            
            # Get priority languages for domain
            priority_languages = self.get_research_priority_languages(domain)
            priority_codes = [lang['code'] for lang in priority_languages]
            
            # Start with highest priority available languages
            selected = []
            for priority_lang in priority_codes:
                if priority_lang in available_languages:
                    selected.append(priority_lang)
                    
                    # Check if we've reached target coverage
                    current_coverage = self.calculate_language_coverage_score(selected, domain)
                    if current_coverage >= target_coverage:
                        break
            
            # If target not reached, add diverse languages
            if len(selected) > 0:
                current_coverage = self.calculate_language_coverage_score(selected, domain)
                
                if current_coverage < target_coverage:
                    # Add languages from different families/scripts
                    selected_families = set()
                    selected_scripts = set()
                    
                    for lang in selected:
                        info = self.get_language_info(lang)
                        selected_families.add(info.get('family'))
                        selected_scripts.add(info.get('script'))
                    
                    # Find diverse candidates
                    for lang in available_languages:
                        if lang not in selected:
                            info = self.get_language_info(lang)
                            family = info.get('family')
                            script = info.get('script')
                            
                            # Add if it increases diversity
                            if (family not in selected_families or 
                                script not in selected_scripts):
                                selected.append(lang)
                                selected_families.add(family)
                                selected_scripts.add(script)
                                
                                # Check coverage again
                                current_coverage = self.calculate_language_coverage_score(selected, domain)
                                if current_coverage >= target_coverage:
                                    break
            
            # Ensure we have at least English if available
            if 'en' not in selected and 'en' in available_languages:
                selected.insert(0, 'en')
            
            # Limit to reasonable number
            selected = selected[:8]  # Max 8 languages
            
            logger.info(f"🎯 Optimized language selection: {selected} (coverage: {self.calculate_language_coverage_score(selected, domain):.2f})")
            return selected
            
        except Exception as e:
            logger.error(f"❌ Language optimization failed: {e}")
            return ['en'] if 'en' in available_languages else available_languages[:3]
    
    def get_regional_language_preferences(self, region: str) -> List[str]:
        """
        Get language preferences for a specific region
        
        Args:
            region: ISO country code or region name
            
        Returns:
            List of preferred languages for the region
        """
        try:
            region_upper = region.upper()
            
            # Find languages commonly used in this region
            regional_languages = []
            
            for lang_code, lang_info in self.language_mappings.items():
                if region_upper in lang_info.get('regions', []):
                    regional_languages.append(lang_code)
            
            # Sort by priority
            regional_languages.sort(
                key=lambda x: self.language_mappings[x].get('priority', 0.0),
                reverse=True
            )
            
            # Add major international languages
            international_langs = ['en', 'zh', 'fr', 'es', 'ar']
            for lang in international_langs:
                if lang not in regional_languages:
                    regional_languages.append(lang)
            
            return regional_languages[:6]  # Top 6 languages
            
        except Exception as e:
            logger.error(f"❌ Failed to get regional preferences for {region}: {e}")
            return ['en', 'zh', 'de', 'fr']
    
    def validate_language_code(self, language_code: str) -> bool:
        """
        Validate if a language code is supported
        
        Args:
            language_code: Language code to validate
            
        Returns:
            True if supported, False otherwise
        """
        try:
            if not language_code or not isinstance(language_code, str):
                return False
            
            code = language_code.lower().strip()
            
            # Check our mappings
            if code in self.language_mappings:
                return True
            
            # Check pycountry
            try:
                lang = pycountry.languages.get(alpha_2=code)
                return lang is not None
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Language code validation failed: {e}")
            return False
    
    def get_language_statistics(self, languages: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a set of languages
        
        Args:
            languages: List of language codes
            
        Returns:
            Dictionary with language statistics
        """
        try:
            if not languages:
                return {'error': 'No languages provided'}
            
            valid_languages = [lang for lang in languages if self.validate_language_code(lang)]
            
            # Basic statistics
            stats = {
                'total_languages': len(languages),
                'valid_languages': len(valid_languages),
                'invalid_languages': len(languages) - len(valid_languages),
                'coverage_score': self.calculate_language_coverage_score(valid_languages)
            }
            
            # Family distribution
            family_distribution = self.group_languages_by_family(valid_languages)
            stats['family_distribution'] = {
                family: len(langs) for family, langs in family_distribution.items()
            }
            
            # Script analysis
            scripts = set()
            priorities = []
            regions = set()
            
            for lang in valid_languages:
                info = self.get_language_info(lang)
                scripts.add(info.get('script', 'Unknown'))
                priorities.append(info.get('priority', 0.5))
                regions.update(info.get('regions', []))
            
            stats['script_diversity'] = len(scripts)
            stats['unique_scripts'] = list(scripts)
            stats['average_priority'] = sum(priorities) / len(priorities) if priorities else 0.0
            stats['regional_coverage'] = len(regions)
            
            # Quality assessment
            high_priority_count = sum(1 for p in priorities if p > 0.8)
            stats['quality_assessment'] = {
                'high_priority_languages': high_priority_count,
                'quality_score': min(1.0, (high_priority_count / len(valid_languages)) * 2) if valid_languages else 0.0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Language statistics calculation failed: {e}")
            return {'error': str(e)}

# Utility functions for backward compatibility
def get_language_info(language_code: str) -> Dict[str, Any]:
    """Get language information"""
    utils = LanguageUtils()
    return utils.get_language_info(language_code)

def get_priority_languages(domain: str = None) -> List[str]:
    """Get priority languages for research"""
    utils = LanguageUtils()
    priority_langs = utils.get_research_priority_languages(domain)
    return [lang['code'] for lang in priority_langs]

def calculate_language_coverage(languages: List[str], domain: str = None) -> float:
    """Calculate language coverage score"""
    utils = LanguageUtils()
    return utils.calculate_language_coverage_score(languages, domain)

