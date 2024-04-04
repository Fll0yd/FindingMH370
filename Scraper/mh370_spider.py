# mh370_spider.py

import logging
import asyncio
from typing import List, Dict, Any
from functools import wraps
import tweepy
import praw
from textblob import TextBlob
from spacy.lang.en import English
from gensim import models
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Other imports...

class MH370Spider:
    def __init__(self):
        self.api_client = APIClient()
        self.nlp = English()
        load_dotenv()

    async def fetch_all_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from all sources."""
        twitter_data = await self.fetch_twitter_data(query)
        reddit_data = await self.fetch_reddit_data(query)
        instagram_data = await self.fetch_instagram_data(query)
        youtube_data = await self.fetch_youtube_data(query)
        
        all_data = twitter_data + reddit_data + instagram_data + youtube_data
        logger.info(f"Fetched {len(all_data)} items from all sources.")
        return all_data

    async def fetch_twitter_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Twitter API."""
        # Implementation...
    
    async def fetch_reddit_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Reddit API."""
        # Implementation...
    
    async def fetch_instagram_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Instagram API."""
        # Implementation...
    
    async def fetch_youtube_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from YouTube API."""
        # Implementation...

# mh370_spider.py

def retry_on_error(max_retries: int = 3, retry_delay: int = 1):
    """Decorator to retry a function on error."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {i + 1} failed: {str(e)}")
                    await asyncio.sleep(retry_delay * (2 ** i))
            raise Exception(f"All retry attempts failed for {func.__name__}.")
        return wrapper
    return decorator

class MH370Spider:
    # Other methods...

    @retry_on_error()
    async def fetch_twitter_data(self, query: str) -> List[Dict[str, Any]]:

    @retry_on_error()
    async def fetch_reddit_data(self, query: str) -> List[Dict[str, Any]]:

    @retry_on_error()
    async def fetch_instagram_data(self, query: str) -> List[Dict[str, Any]]:

    @retry_on_error()
    async def fetch_youtube_data(self, query: str) -> List[Dict[str, Any]]:

# mh370_spider.py

def retry_on_error(max_retries: int = 3, retry_delay: int = 1):
    """Decorator to retry a function on error."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {i + 1} failed: {str(e)}")
                    await asyncio.sleep(retry_delay * (2 ** i))
            raise Exception(f"All retry attempts failed for {func.__name__}.")
        return wrapper
    return decorator

class MH370Spider:
    def __init__(self):
        self.api_client = APIClient()
        self.nlp = English()
        load_dotenv()
        self.logger = logging.getLogger(__name__)

    async def fetch_all_data(self, query: str) -> List[Dict[str, Any]]:

    async def fetch_twitter_data(self, query: str) -> List[Dict[str, Any]]:

    async def fetch_reddit_data(self, query: str) -> List[Dict[str, Any]]:

    async def fetch_instagram_data(self, query: str) -> List[Dict[str, Any]]:

    async def fetch_youtube_data(self, query: str) -> List[Dict[str, Any]]:
