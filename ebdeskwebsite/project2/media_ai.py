import os
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import webbrowser
import re
import networkx as nx
from datetime import datetime
import spacy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import base64
from io import BytesIO
import seaborn as sns

# Download necessary NLTK resources
try:
    for resource in ['vader_lexicon', 'punkt', 'stopwords', 'maxent_ne_chunker', 'words']:
        nltk.download(resource, quiet=True)
except Exception as e:
    print(f"NLTK resource download issue: {e}")

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
except Exception:
    print("spaCy model not available, falling back to NLTK for NER")
    spacy_available = False

class MediaIntelligenceSystem:
    """Streamlined Media Intelligence System with five key analyses"""
    
    def __init__(self, output_dir="output"):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.spacy_available = spacy_available
        self.output_dir = output_dir
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        plt.style.use('ggplot')
        sns.set_palette("deep")
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        if not text: return ""
        text = text.lower()
        text = re.sub(r'http\S+|[^\w\s]|\d+', ' ', text)
        tokens = word_tokenize(text)
        return ' '.join(word for word in tokens if word not in self.stop_words)
    
    def analyze_sentiment_with_score(self, text):
        """Analyze sentiment of text"""
        if not text:
            return {"overall_score": 0, "classification": "neutral", 
                    "positive_score": 0, "negative_score": 0, "neutral_score": 1, "sentence_analysis": []}
        
        # Overall sentiment
        scores = self.sentiment_analyzer.polarity_scores(text)
        result = {
            "overall_score": scores["compound"],
            "classification": "positive" if scores["compound"] >= 0.05 else 
                              "negative" if scores["compound"] <= -0.05 else "neutral",
            "positive_score": scores["pos"],
            "negative_score": scores["neg"],
            "neutral_score": scores["neu"],
            "sentence_analysis": []
        }
        
        # Analyze individual sentences
        for sentence in sent_tokenize(text):
            if not sentence.strip(): continue
            sent_scores = self.sentiment_analyzer.polarity_scores(sentence)
            classification = "positive" if sent_scores["compound"] >= 0.05 else \
                             "negative" if sent_scores["compound"] <= -0.05 else "neutral"
            result["sentence_analysis"].append({
                "text": sentence,
                "score": sent_scores["compound"],
                "classification": classification
            })
        
        # Identify most positive and negative sentences
        if result["sentence_analysis"]:
            sorted_sentences = sorted(result["sentence_analysis"], key=lambda x: x["score"])
            if sorted_sentences[0]["classification"] == "negative":
                result["most_negative"] = sorted_sentences[0]["text"]
            if sorted_sentences[-1]["classification"] == "positive":
                result["most_positive"] = sorted_sentences[-1]["text"]
        
        return result
    
    def plot_sentiment_distribution(self, sentiment_data):
        """Create pie chart of sentiment distribution"""
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [
            sentiment_data['positive_score'],
            sentiment_data['negative_score'],
            sentiment_data['neutral_score']
        ]
        colors = ['#5cb85c', '#d9534f', '#f0ad4e']
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Distribution', fontsize=16)
        
        return self._fig_to_base64()
    
    def plot_sentence_sentiment(self, sentence_analysis):
        """Plot sentiment scores for each sentence"""
        if not sentence_analysis:
            return None
            
        scores = [s['score'] for s in sentence_analysis]
        colors = ['#d9534f' if s < -0.05 else '#f0ad4e' if s < 0.05 else '#5cb85c' for s in scores]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(scores)), scores, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Sentence Index', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.title('Sentiment Score by Sentence', fontsize=16)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def analyze_topics(self, text, custom_topics=None):
        """Identify main topics in text"""
        if not text:
            return {"main_topics": [], "topic_distribution": {}, "key_statements": {}, "quotes": []}
        
        # Default topics if none provided
        topics = custom_topics or [
            "politics", "economy", "business", "technology", "health", 
            "environment", "education", "entertainment", "sports", 
            "science", "society", "culture", "international"
        ]
        
        # Extract quotes
        quotes = re.findall(r'"([^"]*)"', text)
        
        # Topic analysis
        main_topics, topic_distribution = self._analyze_topics_by_keywords(text, topics)
        
        # Find key statements for each topic
        key_statements = {}
        sentences = sent_tokenize(text)
        
        for topic in main_topics:
            statements = [s for s in sentences if re.search(r'\b' + re.escape(topic) + r'\b', s.lower())]
            if statements:
                key_statements[topic] = statements[:3]  # Top 3 statements
        
        return {
            "main_topics": main_topics,
            "topic_distribution": topic_distribution,
            "key_statements": key_statements,
            "quotes": quotes
        }
    
    def _analyze_topics_by_keywords(self, text, topics):
        """Analyze topics using keyword matching"""
        # Keywords for each topic
        topic_keywords = {
            "politics": ["government", "election", "policy", "president", "vote", "political", "democracy"],
            "economy": ["market", "economy", "financial", "trade", "economic", "inflation", "recession"],
            "business": ["company", "business", "corporate", "industry", "CEO", "profit", "startup"],
            "technology": ["tech", "technology", "digital", "software", "hardware", "AI", "internet"],
            "health": ["health", "medical", "doctor", "disease", "treatment", "patient", "healthcare"],
            "environment": ["climate", "environment", "pollution", "sustainable", "renewable", "green"],
            "education": ["school", "education", "student", "teacher", "university", "college", "learning"],
            "entertainment": ["movie", "music", "celebrity", "film", "television", "actor", "actress"],
            "sports": ["game", "team", "player", "sport", "championship", "athlete", "tournament"],
            "science": ["research", "scientist", "discovery", "scientific", "study", "experiment"],
            "society": ["community", "social", "public", "people", "society", "cultural", "demographic"],
            "culture": ["art", "culture", "heritage", "tradition", "cultural", "artist", "museum"],
            "international": ["global", "world", "international", "foreign", "country", "nation", "diplomatic"]
        }
        
        # Count keyword matches
        topic_counts = {topic: 0 for topic in topics}
        
        for topic in topics:
            if topic in topic_keywords:
                for keyword in topic_keywords[topic]:
                    topic_counts[topic] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
        
        # Calculate distribution
        total = sum(topic_counts.values())
        topic_distribution = {t: count/total for t, count in topic_counts.items() if count > 0} if total > 0 else {}
        
        # Get main topics (score > 0.1)
        main_topics = [t for t, score in sorted(topic_distribution.items(), 
                                                key=lambda x: x[1], reverse=True)[:5] if score > 0.1]
        
        return main_topics, topic_distribution
    
    def plot_topic_distribution(self, topic_distribution):
        """Create bar chart of topic distribution"""
        if not topic_distribution:
            return None
        
        # Sort and prepare data
        topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)
        labels = [topic.capitalize() for topic, _ in topics[:10]]
        values = [score for _, score in topics[:10]]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values, color=sns.color_palette("deep", len(labels)))
        
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Relative Frequency', fontsize=12)
        plt.title('Topic Distribution', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')
        
        plt.ylim(0, max(values) * 1.2)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def analyze_5w1h(self, text):
        """Extract Who, What, When, Where, Why, How from text"""
        if not text:
            return {"who": [], "what": [], "when": [], "where": [], "why": [], "how": []}
        
        result = {"who": [], "what": [], "when": [], "where": [], "why": [], "how": []}
        
        # Get entities for Who and Where
        entities = self._extract_entities(text)
        
        # WHO: People and organizations
        if "PER" in entities: result["who"].extend(entities["PER"])
        if "ORG" in entities: result["who"].extend(entities["ORG"])
        
        # WHERE: Locations
        if "LOC" in entities: result["where"].extend(entities["LOC"])
        if "GPE" in entities: result["where"].extend(entities["GPE"])
            
        # WHEN: Time references
        time_patterns = [
            r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),?\s+\d{4}\b',
            r'\b(?:today|yesterday|tomorrow)\b',
            r'\b(?:last|next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\bin\s+\d{4}\b',
            r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        ]
        
        for pattern in time_patterns:
            result["when"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # WHAT: Actions and events
        what_patterns = [
            r'\b(?:announced|launched|released|introduced|developed|created|built|established|formed|founded|started)\s+(?:a|an|the)?\s+\w+(?:\s+\w+){0,5}\b',
            r'\b(?:occurred|happened|took place|began|ended|concluded|finished|started)\b'
        ]
        
        for pattern in what_patterns:
            result["what"].extend([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])
        
        # WHY: Reasons
        why_patterns = [
            r'because\s+(?:\w+\s+){1,15}',
            r'due to\s+(?:\w+\s+){1,15}',
            r'as a result of\s+(?:\w+\s+){1,15}',
            r'in order to\s+(?:\w+\s+){1,15}',
            r'for the purpose of\s+(?:\w+\s+){1,15}',
            r'the reason\s+(?:\w+\s+){1,15}'
        ]
        
        for pattern in why_patterns:
            result["why"].extend([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])
        
        # HOW: Methods
        how_patterns = [
            r'by\s+(?:using|utilizing|employing|applying|implementing)\s+(?:\w+\s+){1,15}',
            r'through\s+(?:\w+\s+){1,15}',
            r'via\s+(?:\w+\s+){1,15}',
            r'with the help of\s+(?:\w+\s+){1,15}',
            r'by means of\s+(?:\w+\s+){1,15}'
        ]
        
        for pattern in how_patterns:
            result["how"].extend([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])
        
        # Clean up results
        for key in result:
            result[key] = list(set(item for item in result[key] if item.strip()))
        
        return result
    
    def plot_5w1h_distribution(self, five_w_one_h):
        """Create bar chart showing 5W1H elements distribution"""
        if not five_w_one_h:
            return None
        
        # Count elements
        counts = {key: len(value) for key, value in five_w_one_h.items()}
        
        # Set colors
        colors = {
            'who': '#3498db', 'what': '#9b59b6', 'when': '#e74c3c',
            'where': '#2ecc71', 'why': '#f39c12', 'how': '#1abc9c'
        }
        
        plt.figure(figsize=(10, 6))
        categories = list(counts.keys())
        values = list(counts.values())
        bar_colors = [colors[cat] for cat in categories]
        
        bars = plt.bar(categories, values, color=bar_colors)
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('5W1H Analysis - Elements Count', fontsize=16)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def analyze_actors(self, text):
        """Analyze relationships between entities in text"""
        if not text:
            return {"actors": [], "relationships": [], "primary_actors": []}
        
        # Extract entities
        entities = self._extract_entities(text)
        
        actors = []
        if "PER" in entities:
            actors.extend([(entity, "person") for entity in entities["PER"]])
        if "ORG" in entities:
            actors.extend([(entity, "organization") for entity in entities["ORG"]])
        
        # Create relationship graph
        G = nx.Graph()
        
        # Add actors as nodes
        for actor, actor_type in actors:
            G.add_node(actor, type=actor_type)
        
        # Find co-occurrences in sentences
        for sentence in sent_tokenize(text):
            sentence_actors = []
            
            for actor, _ in actors:
                if re.search(r'\b' + re.escape(actor) + r'\b', sentence, re.IGNORECASE):
                    sentence_actors.append(actor)
            
            # Create edges between co-occurring actors
            for i, actor1 in enumerate(sentence_actors):
                for actor2 in sentence_actors[i+1:]:
                    if G.has_edge(actor1, actor2):
                        G[actor1][actor2]['weight'] += 1
                    else:
                        G.add_edge(actor1, actor2, weight=1)
        
        # Identify primary actors
        primary_actors = []
        if G.nodes():
            centrality = nx.degree_centrality(G)
            primary_actors = [a for a, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Convert to serializable format
        nodes = [{"id": node, "type": G.nodes[node].get("type", "unknown"), "connections": G.degree(node)} 
                for node in G.nodes()]
        
        edges = [{"source": u, "target": v, "weight": d["weight"]} for u, v, d in G.edges(data=True)]
        
        return {
            "actors": [{"name": actor, "type": actor_type} for actor, actor_type in actors],
            "relationships": edges,
            "primary_actors": primary_actors,
            "graph": G  # For visualization
        }
    
    def plot_actor_network(self, actor_analysis):
        """Create network graph visualization of actor relationships"""
        if not actor_analysis or 'graph' not in actor_analysis or not actor_analysis['graph'].nodes():
            return None
        
        G = actor_analysis['graph']
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 8))
        
        # Node colors and sizes
        node_colors = ['#3498db' if G.nodes[n].get('type', '') == 'person' else '#e74c3c' for n in G.nodes()]
        centrality = nx.degree_centrality(G)
        node_sizes = [1000 * centrality[n] + 100 for n in G.nodes()]
        
        # Edge weights
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title('Actor Relationship Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def analyze_locations(self, text):
        """Analyze locations mentioned in text"""
        if not text:
            return {"locations": [], "frequency": {}, "primary_locations": []}
        
        # Extract locations
        entities = self._extract_entities(text)
        
        locations = []
        if "LOC" in entities:
            locations.extend(entities["LOC"])
        if "GPE" in entities:
            locations.extend(entities["GPE"])
        
        # Count frequency
        location_counts = Counter(locations)
        
        # Calculate percentages
        total = sum(location_counts.values())
        frequency = {}
        
        if total > 0:
            for location, count in location_counts.items():
                frequency[location] = {
                    "count": count,
                    "percentage": (count / total) * 100
                }
        
        # Find primary locations
        primary_locations = [loc for loc, _ in sorted(location_counts.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5]]
        
        # Find contexts for top locations
        contexts = {}
        sentences = sent_tokenize(text)
        
        for location in primary_locations:
            location_sentences = [s for s in sentences 
                                if re.search(r'\b' + re.escape(location) + r'\b', s, re.IGNORECASE)]
            if location_sentences:
                contexts[location] = location_sentences[:3]
        
        return {
            "locations": locations,
            "frequency": frequency,
            "primary_locations": primary_locations,
            "contexts": contexts
        }
    
    def plot_location_distribution(self, location_analysis):
        """Create horizontal bar chart of location distribution"""
        if not location_analysis or not location_analysis.get('frequency'):
            return None
        
        # Get top 10 locations
        frequency = location_analysis['frequency']
        locations = sorted(frequency.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        
        names = [loc.capitalize() for loc, _ in locations]
        counts = [data['count'] for _, data in locations]
        
        # Create chart
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(names))
        
        bars = plt.barh(y_pos, counts, align='center', 
                      color=sns.color_palette("deep", len(locations)))
        plt.yticks(y_pos, names)
        
        # Add labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            percentage = frequency[locations[i][0]]['percentage']
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)} ({percentage:.1f}%)', va='center')
        
        plt.xlabel('Mentions', fontsize=12)
        plt.title('Location Mentions Distribution', fontsize=16)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _extract_entities(self, text):
        """Extract named entities using available NLP tools"""
        entities = defaultdict(list)
        
        # Try spaCy first if available
        if self.spacy_available:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["PER"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["ORG"].append(ent.text)
                elif ent.label_ in ("GPE", "LOC"):
                    entities["LOC"].append(ent.text)
                    
        # Fallback to NLTK if needed
        if not any(entities.values()):
            for sentence in nltk.sent_tokenize(text):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join(c[0] for c in chunk)
                        if chunk.label() == 'PERSON':
                            entities["PER"].append(entity_text)
                        elif chunk.label() == 'ORGANIZATION':
                            entities["ORG"].append(entity_text)
                        elif chunk.label() == 'GPE':
                            entities["LOC"].append(entity_text)
        
        # Remove duplicates and normalize
        for entity_type in entities:
            entities[entity_type] = list(set(e for e in entities[entity_type] if len(e) >= 2))
        
        return dict(entities)
    
    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_interactive_html_analysis(self, text, title="Media Analysis Report"):
        """Generate HTML report with visualizations"""
        if not text:
            return "No content to analyze"
        
        # Perform all analyses
        sentiment_analysis = self.analyze_sentiment_with_score(text)
        topic_analysis = self.analyze_topics(text)
        five_w_one_h = self.analyze_5w1h(text)
        actors_analysis = self.analyze_actors(text)
        location_analysis = self.analyze_locations(text)
        
        # Generate visualizations
        sentiment_pie_chart = self.plot_sentiment_distribution(sentiment_analysis)
        sentiment_sentence_chart = self.plot_sentence_sentiment(sentiment_analysis.get('sentence_analysis', []))
        topic_chart = self.plot_topic_distribution(topic_analysis.get('topic_distribution', {}))
        five_w_one_h_chart = self.plot_5w1h_distribution(five_w_one_h)
        actor_network_chart = self.plot_actor_network(actors_analysis)
        location_chart = self.plot_location_distribution(location_analysis)
        
        # Create HTML report (simplified)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                header h1 {{
                    color: white;
                    margin: 0;
                }}
                .timestamp {{
                    color: #bdc3c7;
                    font-size: 0.9em;
                    margin-top: 10px;
                }}
                section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .positive {{color: #2ecc71;}}
                .negative {{color: #e74c3c;}}
                .neutral {{color: #f39c12;}}
                .card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    background-color: #f8f9fa;
                }}
                .quote {{
                    font-style: italic;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin: 15px 0;
                }}
                .flex-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .flex-item {{
                    flex: 1;
                    min-width: 300px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>{title}</h1>
                <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</div>
            </header>
            
            <section id="summary">
                <h2>Executive Summary</h2>
                <p>Overall Sentiment: <strong class="{sentiment_analysis['classification']}">{sentiment_analysis['classification'].capitalize()}</strong> (Score: {sentiment_analysis['overall_score']:.2f})</p>
                <p>Main Topics: <strong>{', '.join(topic_analysis['main_topics'][:3])}</strong></p>
                <p>Primary Actors: <strong>{', '.join(actors_analysis['primary_actors'][:3])}</strong></p>
                <p>Main Locations: <strong>{', '.join(location_analysis['primary_locations'][:3])}</strong></p>
            </section>
            
            <section id="sentiment">
                <h2>Sentiment Analysis</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        <div class="chart">
                            <h3>Sentiment Distribution</h3>
                            <img src="{sentiment_pie_chart}" alt="Sentiment Distribution">
                        </div>
                    </div>
                    <div class="flex-item">
                        <div class="chart">
                            <h3>Sentiment by Sentence</h3>
                            <img src="{sentiment_sentence_chart}" alt="Sentiment by Sentence">
                        </div>
                    </div>
                </div>
        """
        
        # Add most positive and negative sentences if available
        if 'most_positive' in sentiment_analysis:
            html += f"""
                <div class="card">
                    <h3>Most Positive Statement</h3>
                    <p class="quote positive">{sentiment_analysis['most_positive']}</p>
                </div>
            """
        
        if 'most_negative' in sentiment_analysis:
            html += f"""
                <div class="card">
                    <h3>Most Negative Statement</h3>
                    <p class="quote negative">{sentiment_analysis['most_negative']}</p>
                </div>
            """
        
        # Topics section
        html += f"""
            </section>
            
            <section id="topics">
                <h2>Topic Analysis</h2>
                <div class="chart">
                    <h3>Topic Distribution</h3>
                    <img src="{topic_chart}" alt="Topic Distribution Chart">
                </div>
                
                <h3>Key Statements by Topic</h3>
        """
        
        # Add key statements for each main topic
        for topic in topic_analysis['main_topics']:
            if topic in topic_analysis['key_statements'] and topic_analysis['key_statements'][topic]:
                html += f"""
                    <div class="card">
                        <h4>{topic.capitalize()}</h4>
                """
                for statement in topic_analysis['key_statements'][topic]:
                    html += f'<p class="quote">{statement}</p>'
                html += '</div>'
        
        # Add quotes if available
        if topic_analysis['quotes']:
            html += f"""
                <h3>Notable Quotes</h3>
                <div class="card">
            """
            for quote in topic_analysis['quotes'][:5]:  # Show top 5 quotes
                html += f'<p class="quote">"{quote}"</p>'
            html += '</div>'
        
        # 5W1H section
        html += f"""
            </section>
            
            <section id="five-w-one-h">
                <h2>5W1H Analysis</h2>
                <div class="chart">
                    <img src="{five_w_one_h_chart}" alt="5W1H Distribution Chart">
                </div>
                
                <div class="flex-container">
        """
        
        # Who
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>Who</h3>
        """
        if five_w_one_h['who']:
            for entity in five_w_one_h['who'][:10]:  # Show top 10
                html += f'<p>{entity}</p>'
        else:
            html += '<p>No specific individuals or organizations identified.</p>'
        html += '</div></div>'
        
        # What
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>What</h3>
        """
        if five_w_one_h['what']:
            for action in five_w_one_h['what'][:10]:  # Show top 10
                html += f'<p>{action}</p>'
        else:
            html += '<p>No specific actions or events identified.</p>'
        html += '</div></div>'
        
        # When
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>When</h3>
        """
        if five_w_one_h['when']:
            for time in five_w_one_h['when'][:10]:  # Show top 10
                html += f'<p>{time}</p>'
        else:
            html += '<p>No specific time references identified.</p>'
        html += '</div></div>'
        
        # Where
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>Where</h3>
        """
        if five_w_one_h['where']:
            for place in five_w_one_h['where'][:10]:  # Show top 10
                html += f'<p>{place}</p>'
        else:
            html += '<p>No specific locations identified.</p>'
        html += '</div></div>'
        
        # Why
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>Why</h3>
        """
        if five_w_one_h['why']:
            for reason in five_w_one_h['why'][:10]:  # Show top 10
                html += f'<p>{reason}</p>'
        else:
            html += '<p>No specific reasons or motivations identified.</p>'
        html += '</div></div>'
        
        # How
        html += f"""
            <div class="flex-item">
                <div class="card">
                    <h3>How</h3>
        """
        if five_w_one_h['how']:
            for method in five_w_one_h['how'][:10]:  # Show top 10
                html += f'<p>{method}</p>'
        else:
            html += '<p>No specific methods or processes identified.</p>'
        html += '</div></div>'
        
        # Actor Analysis section
        html += f"""
            </div>
            </section>
            
            <section id="actors">
                <h2>Actor Analysis</h2>
                <div class="chart">
                    <h3>Actor Network</h3>
                    <img src="{actor_network_chart}" alt="Actor Network Chart">
                </div>
                
                <h3>Primary Actors</h3>
                <div class="card">
        """
        
        if actors_analysis['primary_actors']:
            for actor in actors_analysis['primary_actors']:
                # Find type of actor
                actor_type = "unknown"
                for a in actors_analysis['actors']:
                    if a['name'] == actor:
                        actor_type = a['type']
                        break
                html += f'<p><strong>{actor}</strong> (Type: {actor_type})</p>'
        else:
            html += '<p>No primary actors identified.</p>'
        
        html += '</div>'
        
        # Location Analysis section
        html += f"""
            </section>
            
            <section id="locations">
                <h2>Location Analysis</h2>
                <div class="chart">
                    <h3>Location Distribution</h3>
                    <img src="{location_chart}" alt="Location Distribution Chart">
                </div>
                
                <h3>Primary Locations Context</h3>
        """
        
        if location_analysis['contexts']:
            for location, contexts in location_analysis['contexts'].items():
                html += f"""
                    <div class="card">
                        <h4>{location}</h4>
                """
                for context in contexts:
                    html += f'<p class="quote">{context}</p>'
                html += '</div>'
        else:
            html += '<p>No location contexts identified.</p>'
        
        # Close HTML
        html += """
            </section>
            
            <footer>
                <p>Generated by Media Intelligence System</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report to file
        file_path = os.path.join(self.output_dir, f"media_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return {
            "html_report": html,
            "file_path": file_path,
            "sentiment_analysis": sentiment_analysis,
            "topic_analysis": topic_analysis,
            "five_w_one_h": five_w_one_h,
            "actors_analysis": actors_analysis,
            "location_analysis": location_analysis
        }
    
    def analyze_text_document(self, text, title="Document Analysis", custom_topics=None):
        """Complete analysis of a text document with all five analyses"""
        if not text or len(text.strip()) == 0:
            return {"error": "No text provided for analysis"}
        
        # Create analysis result object
        result = {
            "title": title,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "text_length": len(text),
            "sentence_count": len(sent_tokenize(text)),
            "word_count": len(word_tokenize(text)),
            "analyses": {}
        }
        
        # Perform all five analyses
        result["analyses"]["sentiment"] = self.analyze_sentiment_with_score(text)
        result["analyses"]["topics"] = self.analyze_topics(text, custom_topics)
        result["analyses"]["5w1h"] = self.analyze_5w1h(text)
        result["analyses"]["actors"] = self.analyze_actors(text)
        result["analyses"]["locations"] = self.analyze_locations(text)
        
        # Generate HTML report
        report_result = self.generate_interactive_html_analysis(text, title)
        result["html_report_path"] = report_result["file_path"]
        
        return result
    
    def generate_text_summary(self, text, max_length=300):
        """Generate a brief summary of the text based on key sentences"""
        if not text:
            return "No text provided for summarization."
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text  # Already short enough
        
        # Extract key information from analyses
        sentiment = self.analyze_sentiment_with_score(text)
        topics = self.analyze_topics(text)
        actors = self.analyze_actors(text)
        locations = self.analyze_locations(text)
        
        # Score each sentence based on presence of key information
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Check for main topics
            for topic in topics["main_topics"]:
                if re.search(r'\b' + re.escape(topic) + r'\b', sentence.lower()):
                    score += 2
            
            # Check for primary actors
            for actor in actors["primary_actors"]:
                if re.search(r'\b' + re.escape(actor) + r'\b', sentence):
                    score += 2
            
            # Check for primary locations
            for location in locations["primary_locations"]:
                if re.search(r'\b' + re.escape(location) + r'\b', sentence):
                    score += 1
            
            # Give more weight to first and last sentences
            if i == 0 or i == len(sentences) - 1:
                score += 1
            
            # Give more weight to sentences with strong sentiment
            sent_sentiment = self.sentiment_analyzer.polarity_scores(sentence)
            if abs(sent_sentiment["compound"]) > 0.5:
                score += 1
            
            sentence_scores[i] = score
        
        # Select top sentences
        top_sentence_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top sentences (but keep original order)
        num_sentences = min(5, max(3, len(sentences) // 5))  # Adjust number of sentences based on text length
        selected_indices = [idx for idx, score in top_sentence_indices[:num_sentences]]
        selected_indices.sort()
        
        # Build summary with selected sentences
        summary = " ".join([sentences[idx] for idx in selected_indices])
        
        # If still too long, truncate with ellipsis
        if len(summary) > max_length:
            words = summary.split()
            truncated = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_length - 3:  # Account for ellipsis
                    break
                truncated.append(word)
                current_length += len(word) + 1
            
            summary = " ".join(truncated) + "..."
            
        return summary
    
    def compare_texts(self, text1, text2, title1="Text 1", title2="Text 2"):
        """Compare two texts and identify similarities and differences"""
        if not text1 or not text2:
            return {"error": "Both texts are required for comparison"}
        
        # Analyze both texts
        analysis1 = {
            "sentiment": self.analyze_sentiment_with_score(text1),
            "topics": self.analyze_topics(text1),
            "actors": self.analyze_actors(text1),
            "locations": self.analyze_locations(text1)
        }
        
        analysis2 = {
            "sentiment": self.analyze_sentiment_with_score(text2),
            "topics": self.analyze_topics(text2),
            "actors": self.analyze_actors(text2),
            "locations": self.analyze_locations(text2)
        }
        
        # Compare results
        comparison = {
            "titles": {
                "text1": title1,
                "text2": title2
            },
            "sentiment": {
                "text1": analysis1["sentiment"]["classification"],
                "text1_score": analysis1["sentiment"]["overall_score"],
                "text2": analysis2["sentiment"]["classification"],
                "text2_score": analysis2["sentiment"]["overall_score"],
                "difference": analysis2["sentiment"]["overall_score"] - analysis1["sentiment"]["overall_score"]
            },
            "topics": {
                "common": list(set(analysis1["topics"]["main_topics"]).intersection(set(analysis2["topics"]["main_topics"]))),
                "unique_to_text1": list(set(analysis1["topics"]["main_topics"]) - set(analysis2["topics"]["main_topics"])),
                "unique_to_text2": list(set(analysis2["topics"]["main_topics"]) - set(analysis1["topics"]["main_topics"]))
            },
            "actors": {
                "common": list(set(analysis1["actors"]["primary_actors"]).intersection(set(analysis2["actors"]["primary_actors"]))),
                "unique_to_text1": list(set(analysis1["actors"]["primary_actors"]) - set(analysis2["actors"]["primary_actors"])),
                "unique_to_text2": list(set(analysis2["actors"]["primary_actors"]) - set(analysis1["actors"]["primary_actors"]))
            },
            "locations": {
                "common": list(set(analysis1["locations"]["primary_locations"]).intersection(set(analysis2["locations"]["primary_locations"]))),
                "unique_to_text1": list(set(analysis1["locations"]["primary_locations"]) - set(analysis2["locations"]["primary_locations"])),
                "unique_to_text2": list(set(analysis2["locations"]["primary_locations"]) - set(analysis1["locations"]["primary_locations"]))
            }
        }
        
        return comparison
    
    def analyze_bias(self, text):
        """Detect potential bias in text based on language analysis"""
        if not text:
            return {"bias_score": 0, "indicators": []}
        
        # List of loaded terms that may indicate bias
        bias_indicators = {
            "political": ["leftist", "rightist", "liberal", "conservative", "radical", "extremist", 
                         "socialist", "communist", "fascist", "patriot", "traitor"],
            "emotional": ["disgusting", "outrageous", "terrific", "horrible", "amazing", "appalling",
                         "catastrophic", "wonderful", "terrible", "disastrous"],
            "exaggeration": ["always", "never", "every", "all", "none", "absolutely", "completely",
                           "totally", "undoubtedly", "unquestionably"],
            "certainty": ["certainly", "definitely", "undoubtedly", "obviously", "clearly",
                         "unquestionably", "undeniably", "indisputably"],
            "generalization": ["everyone knows", "people say", "it is said", "many believe",
                             "experts agree", "studies show", "research proves"]
        }
        
        # Count occurrences of biased language
        results = {
            "indicators": [],
            "categories": {cat: [] for cat in bias_indicators}
        }
        
        # Check for bias indicators
        text_lower = text.lower()
        
        for category, terms in bias_indicators.items():
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    results["categories"][category].append({
                        "term": term,
                        "count": len(matches)
                    })
                    results["indicators"].extend([{
                        "term": term,
                        "category": category
                    } for _ in range(len(matches))])
        
        # Calculate bias score (0-1 scale)
        word_count = len(text.split())
        indicator_count = len(results["indicators"])
        
        if word_count > 0:
            normalized_score = min(1.0, (indicator_count / word_count) * 100)
            results["bias_score"] = normalized_score
        else:
            results["bias_score"] = 0
        
        # Bias level classification
        if results["bias_score"] < 0.05:
            results["bias_level"] = "low"
        elif results["bias_score"] < 0.15:
            results["bias_level"] = "moderate"
        else:
            results["bias_level"] = "high"
        
        return results
    
    def save_results_to_json(self, results, filename=None):
        """Save analysis results to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f"analysis_results_{timestamp}.json")
        
        # Create a serializable copy of the results (remove graph objects)
        serializable_results = {}
        
        def clean_for_json(obj):
            """Remove non-serializable objects"""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() 
                        if k != 'graph' and not isinstance(v, nx.Graph)}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = clean_for_json(results)
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        return filename

