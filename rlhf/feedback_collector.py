"""
Module for collecting and storing human feedback for RLHF.
"""

import logging
import json
import os
import sqlite3
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import config

logger = logging.getLogger(__name__)

class FeedbackCollector:
    """
    Collects and stores human feedback for use in RLHF.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the feedback collector.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or config.FEEDBACK_DB_PATH
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self._init_db()
        
        logger.info(f"Initialized FeedbackCollector with database at {self.db_path}")
    
    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            rating REAL NOT NULL,
            feedback_text TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS content (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_feedback(self,
                        content_id: str,
                        rating: float,
                        feedback_text: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Collect and store feedback.
        
        Args:
            content_id: The ID of the content being rated
            rating: Numerical rating (typically 1-5 or 1-10)
            feedback_text: Optional textual feedback
            metadata: Optional additional metadata
            
        Returns:
            The feedback ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            '''
            INSERT INTO feedback 
            (content_id, rating, feedback_text, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (content_id, rating, feedback_text, timestamp, metadata_json)
        )
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Collected feedback (id={feedback_id}) for content {content_id}")
        return feedback_id
    
    def store_content(self,
                     content_id: str,
                     prompt: str,
                     content: str,
                     content_type: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Store content for future reference and feedback collection.
        
        Args:
            content_id: Unique identifier for the content
            prompt: The prompt that generated the content
            content: The generated content
            content_type: Optional type of content
            metadata: Optional additional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            '''
            INSERT OR REPLACE INTO content
            (id, prompt, content, content_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (content_id, prompt, content, content_type, timestamp, metadata_json)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored content with id {content_id}")
    
    def get_feedback(self, content_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve feedback, optionally filtered by content ID.
        
        Args:
            content_id: Optional content ID to filter by
            
        Returns:
            List of feedback entries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        if content_id:
            cursor.execute(
                "SELECT * FROM feedback WHERE content_id = ? ORDER BY timestamp DESC",
                (content_id,)
            )
        else:
            cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
        
        results = [{key: row[key] for key in row.keys()} for row in cursor.fetchall()]
        
        # Parse metadata JSON
        for result in results:
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
        
        conn.close()
        return results
    
    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve content by ID.
        
        Args:
            content_id: The content ID
            
        Returns:
            Content entry or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM content WHERE id = ?", (content_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        result = {key: row[key] for key in row.keys()}
        
        # Parse metadata JSON
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        
        conn.close()
        return result
    
    def get_feedback_pairs(self, 
                          min_rating_diff: float = 1.0, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get pairs of content with significant rating differences for training.
        
        Args:
            min_rating_diff: Minimum rating difference to consider
            limit: Maximum number of pairs to return
            
        Returns:
            List of preferred/dispreferred content pairs
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get content with multiple ratings that have significant differences
        cursor.execute('''
            SELECT 
                c.id,
                c.prompt,
                c.content,
                c.content_type,
                AVG(f.rating) as avg_rating,
                COUNT(f.id) as feedback_count
            FROM content c
            JOIN feedback f ON c.id = f.content_id
            GROUP BY c.id
            HAVING COUNT(f.id) > 1
            ORDER BY feedback_count DESC, avg_rating DESC
            LIMIT ?
        ''', (limit,))
        
        content_ratings = [{key: row[key] for key in row.keys()} for row in cursor.fetchall()]
        
        # Find pairs of content with the same prompt but different ratings
        pairs = []
        processed_prompts = set()
        
        for i, item1 in enumerate(content_ratings):
            prompt = item1['prompt']
            
            if prompt in processed_prompts:
                continue
                
            # Find all content with the same prompt
            same_prompt_items = [
                item2 for item2 in content_ratings 
                if item2['prompt'] == prompt and item2['id'] != item1['id']
            ]
            
            # Sort by rating
            items = [item1] + same_prompt_items
            items.sort(key=lambda x: x['avg_rating'], reverse=True)
            
            # Create pairs with significant rating differences
            for j in range(len(items) - 1):
                for k in range(j + 1, len(items)):
                    rating_diff = items[j]['avg_rating'] - items[k]['avg_rating']
                    if rating_diff >= min_rating_diff:
                        pairs.append({
                            'preferred': {
                                'id': items[j]['id'],
                                'content': items[j]['content'],
                                'rating': items[j]['avg_rating']
                            },
                            'dispreferred': {
                                'id': items[k]['id'],
                                'content': items[k]['content'],
                                'rating': items[k]['avg_rating']
                            },
                            'prompt': prompt,
                            'rating_diff': rating_diff
                        })
            
            processed_prompts.add(prompt)
            
            if len(pairs) >= limit:
                break
        
        conn.close()
        return pairs[:limit]
    
    def export_feedback_data(self, output_file: str):
        """
        Export all feedback data to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all content
        cursor.execute("SELECT * FROM content")
        content_rows = cursor.fetchall()
        content_data = [{key: row[key] for key in row.keys()} for row in content_rows]
        
        # Get all feedback
        cursor.execute("SELECT * FROM feedback")
        feedback_rows = cursor.fetchall()
        feedback_data = [{key: row[key] for key in row.keys()} for row in feedback_rows]
        
        # Parse JSON fields
        for item in content_data:
            if item.get('metadata'):
                item['metadata'] = json.loads(item['metadata'])
                
        for item in feedback_data:
            if item.get('metadata'):
                item['metadata'] = json.loads(item['metadata'])
        
        # Create export data
        export_data = {
            'content': content_data,
            'feedback': feedback_data,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        conn.close()
        logger.info(f"Exported feedback data to {output_file}")