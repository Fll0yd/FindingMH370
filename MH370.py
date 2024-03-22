import asyncio
import cProfile
from logging import configure_logging
import os
import pstats
import shutil
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, patch
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import gensim
import praw
import requests
import schedule
import spacy
import tweepy
import yaml
import json
from  peewee import CharField, Model, TextField
from bs4 import BeautifulSoup
from googleapiclient import discovery
from instagram_private_api import Client
import pymongo
from pymongo import MongoClient
from requests.exceptions import RequestException, Timeout
from scrapy import Selector
from scrapy.exceptions import NotConfigured
from textblob import TextBlob
import tkinter as tk
from tkinter import ttk
from scraper_module import MH370WebScraper
from instagram_module import InstagramAPI
from youtube_module import YouTubeAPI
import logging  

# Configure logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration settings from config.yaml
def load_config(filename: str) -> Dict[str, Any]:
    """Load configuration settings from YAML file."""
    try:
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error("Config file not found.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f'Error parsing YAML file: {e}')
        return {}

config = load_config('config.yaml')

# Set timeouts and buffer sizes for network operations
NETWORK_TIMEOUT = 10  # seconds
BUFFER_SIZE = 8192  # bytes
MAX_RETRIES = 5  # Replace 5 with the number of maximum retries you want

# Set MongoDB connection parameters from config
MONGODB_URI = config.get('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB = config.get('MONGODB_DB', 'mh370')

# Connect to MongoDB
client = MongoClient (MONGODB_URI)
db = client[MONGODB_DB]
collection = db['data']

# Insert report into MongoDB collection
report = {
  'time': '2014-03-08 02:40',
  'location': {
    'latitude': 6.92, 
    'longitude': 103.20
  },
  'altitude': 35000,
  'speed': 471,
  'fuel': 23000
}
collection.insert_one(report)

# Query reports from MongoDB
for report in collection.find():
  print(report['time'], report['location']['latitude'], report['location']['longitude'])

# Update report in MongoDB
filter = {'time': '2014-03-08 02:40'}
update = {'$set': {'altitude': 36000}}
collection.update_one(filter, update)

# Delete report from MongoDB
collection.delete_one({'time': '2014-03-08 02:40'})

class MH370Data(Model):
    """Optimized model for storing MH370-related data."""
    title = TextField(index=True)
    link = TextField()
    snippet = TextField()
    image_urls = TextField()
    video_urls = TextField()
    source = CharField(index=True)
    sentiment = CharField()
    entities = TextField()
    topics = TextField()
    
    class Meta:
        database = db

def migrate_data() -> None:
    """Migrate data from the old schema to the new schema."""
    try:
        old_data = list(MH370Data.select())
        logger.info(f"Found {len(old_data)} items in MH370Data.")
        with db.atomic():
            for item in old_data:
                MH370Data.create(
                    title=item.title,
                    link=item.link,
                    snippet=item.snippet,
                    image_urls=item.image_urls,
                    video_urls=item.video_urls,
                    source=item.source,
                    sentiment=item.sentiment,
                    entities=item.entities,
                    topics=item.topics
                )
        logger.info("Data migration completed.")
    except Exception as e:
        logger.error(f'Error migrating data: {e}')
        logger.exception("Exception occurred during data migration.")

# Function to initiate data migration process
def initiate_migration(old_schema_data: List[Dict[str, Any]]) -> bool:
    """Initiate the data migration process."""
    try:
        with db.atomic():
            for item in old_schema_data:
                MH370Data.create(
                    title=item['title'],
                    link=item['link'],
                    snippet=item['snippet'],
                    image_urls=item['image_urls'],
                    video_urls=item['video_urls'],
                    source=item['source'],
                    sentiment=item['sentiment'],
                    entities=item['entities'],
                    topics=item['topics']
                )
        logger.info("Migration to the new schema completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

def check_new_schema_data() -> bool:
    """Check if data exists in the new schema."""
    try:
        return MH370Data.select().exists()
    except Exception as e:
        logger.error(f"Error while checking new schema data: {e}")
        return False

def is_valid_data(data: Dict[str, Any]) -> bool:
    """Check if the data is valid."""
    return all(key in data for key in ['title', 'link', 'snippet', 'source'])

# Implement database backup mechanism
def backup_database() -> None:
    """Create a backup of the database."""
    try:
        backup_path = 'mh370__backup.db'
        shutil.copyfile('mh370_.db', backup_path)
        logger.info("Database backup created successfully.")
    except Exception as e:
        logger.error(f"Error during database backup: {e}")

def schedule_backup() -> None:
    """Schedule backup to run daily at midnight."""
    schedule.every().day.at("00:00").do(backup_database)
    while True:
        schedule.run_pending()
        time.sleep(1)
class DataMonitorApp:
    def __init__(self, root: tk.Tk):
        """Initialize the DataMonitorApp."""
        self.root = root
        self.root.title("Data Monitor")
        self.tree = ttk.Treeview(self.root, columns=("Title", "Source", "Sentiment"))
        self.tree.heading("#0", text="ID")
        self.tree.heading("Title", text="Title")
        self.tree.heading("Source", text="Source")
        self.tree.heading("Sentiment", text="Sentiment")
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)
        refresh_button = ttk.Button(self.root, text="Refresh Data", command=self.refresh_data)
        refresh_button.grid(row=1, column=0, pady=5)
        add_button = ttk.Button(self.root, text="Add Data", command=self.add_data)
        add_button.grid(row=2, column=0, pady=5)
        delete_button = ttk.Button(self.root, text="Delete Selected", command=self.delete_selected)
        delete_button.grid(row=3, column=0, pady=5)
        self.data = [
            {"id": 1, "title": "Example Title 1", "source": "Twitter", "sentiment": "Positive"},
            {"id": 2, "title": "Example Title 2", "source": "Reddit", "sentiment": "Neutral"},
            {"id": 3, "title": "Example Title 3", "source": "Google", "sentiment": "Negative"},
            {"id": 4, "title": "Example Title 3", "source": "YouTube", "sentiment": "Negative"},
            {"id": 5, "title": "Example Title 3", "source": "Facebook", "sentiment": "Negative"},
        ]
        self.populate_treeview()
    
    def populate_treeview(self):
        """Populate the treeview with data."""
        self.tree.delete(*self.tree.get_children())
        for item in self.data:
            self.tree.insert("", "end", text=item["id"], values=(item["title"], item["source"], item["sentiment"]))
    
    def refresh_data(self):
        """Refresh the data in the treeview."""
        import random
        for item in self.data:
            item["sentiment"] = random.choice(["Positive", "Neutral", "Negative"])
        self.populate_treeview()
    
    def add_data(self):
        """Add new data to the treeview."""
        import random
        new_entry = {
            "id": len(self.data) + 1,
            "title": f"New Title {len(self.data) + 1}",
            "source": random.choice(["Twitter", "Reddit", "Google"]),
            "sentiment": random.choice(["Positive", "Neutral", "Negative"])
        }
        self.data.append(new_entry)
        self.populate_treeview()
    
    def delete_selected(self):
        """Delete the selected data from the treeview."""
        selection = self.tree.selection()
        if selection:
            item_id = self.tree.item(selection[0])["text"]
            for item in self.data:
                if item["id"] == int(item_id):
                    self.data.remove(item)
                    break
            self.populate_treeview()

class APIClient:
    def __init__(self):
        """Initialize the API clients for different platforms."""
        self.twitter_api = self.get_api_client('twitter')
        self.reddit_api = self.get_api_client('reddit')
        self.instagram_api = self.get_api_client('instagram')
        self.youtube_api = self.get_api_client('youtube')

    def get_api_client(self, platform: str):
        """Get the authenticated API client for the specified platform."""
        if platform == 'twitter':
            auth = tweepy.OAuth1UserHandler(
                self.config.get('TWITTER_API_KEY'),
                self.config.get('TWITTER_API_SECRET'),
                self.config.get('TWITTER_ACCESS_TOKEN'),
                self.config.get('TWITTER_ACCESS_TOKEN_SECRET')
            )
            return tweepy.API(auth, timeout=NETWORK_TIMEOUT)
        elif platform == 'reddit':
            return praw.Reddit(
                client_id=self.config.get('REDDIT_CLIENT_ID'),
                client_secret=self.config.get('REDDIT_CLIENT_SECRET'),
                user_agent=self.config.get('REDDIT_USER_AGENT')
            )
        elif platform == 'instagram':
            return InstagramAPI(
                self.config.get('INSTAGRAM_USERNAME'),
                self.config.get('INSTAGRAM_PASSWORD')
            )
        elif platform == 'youtube':
            return YouTubeAPI(self.config.get('YOUTUBE_API_KEY'))
        else:
            raise ValueError(f"Unsupported platform: {platform}")

class MH370Spider:
    def __init__(self):
        """Initialize the spider."""
        configure_logging()  # Configure logging
        self.logger = logging.getLogger(__name__)  # Get logger for the current module
        self.logger.info("Initialized the spider.")
        self.api_client = APIClient()
        # Initialize NLP libraries
        self.nlp = spacy.load('en_core_web_sm')
        self.model = gensim.models.LdaModel.load('topic_model')
        # Initialize the web scraper
        self.scraper = MH370WebScraper()
        self.instagram_data = InstagramAPI()
        self.youtube_data = YouTubeAPI()

    async def parse(self, response: Any) -> None:
        """Parse the response and fetch data from different platforms."""
        try:
            google_data = await self.parse_google(response)
            await self.save_data(google_data)
            self.logger.info("Parsed Google search results and saved data to database.")
            tasks = [
                self.fetch_twitter_data('MH370'),
                self.fetch_reddit_data('MH370'),
                self.fetch_instagram_data('MH370'),  # Add Instagram data fetching task
                self.fetch_youtube_data('MH370')     # Add YouTube data fetching task
            ]
            results = await asyncio.gather(*tasks)
            for result in results:
                await self.save_data(result)
            self.logger.info("Fetched data from Twitter, Reddit, Instagram, and YouTube APIs and saved to database.")
        except Exception as e:
            self.logger.error(f'Error in parsing or fetching data: {str(e)}')
            self.logger.exception("Exception occurred during parsing or fetching data.")

    async def parse_google(self, response: Any) -> List[Dict[str, Any]]:
        """Parse the Google search results."""
        google_data = []
        search_results = response.css('div.tF2Cxc')
        for result in search_results:
            try:
                title = result.css('h3::text').get()
                link = result.css('a::attr(href)').get()
                snippet = result.css('div.IsZvec::text').get()
                image_urls = result.css('img::attr(src)').getall()
                video_urls = result.css('a[href*=watch]::attr(href)').getall()

                # Perform sentiment analysis
                sentiment = TextBlob(snippet).sentiment.polarity

                # Perform entity recognition
                doc = self.nlp(snippet)
                entities = [ent.text for ent in doc.ents]

                # Perform topic modeling
                topics = self.model[doc]

                google_data.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                    'image_urls': image_urls,
                    'video_urls': video_urls,
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            except NotConfigured as e:
                self.logger.error(f'Error in parsing Google data: {str(e)}')
                self.logger.exception("Exception occurred during parsing Google data.")
            except Exception as e:
                self.logger.error(f'Unexpected error in parsing Google data: {str(e)}')
                self.logger.exception("Unexpected exception occurred during parsing Google data.")
        return google_data
    
    async def save_data(self, data: List[Dict[str, Any]]) -> None:
        """Save the data to the database."""
        try:
            if data:
                with db.atomic():
                    for item in data:
                        if self.is_valid_data(item):
                            MH370Data.create(
                                title=item['title'],
                                link=item['link'],
                                snippet=item['snippet'],
                                image_urls=','.join(item['image_urls']),
                                video_urls=','.join(item['video_urls']),
                                source=item['source'],
                                sentiment=item['sentiment'],
                                entities=','.join(item['entities']),
                                topics=','.join(item['topics'])
                            )
                self.logger.info("Saved data to MongoDB database.")
            else:
                self.logger.warning("No data to save.")
        except Exception as e:
            self.logger.error(f'Error saving data to MongoDB database: {str(e)}')
            self.logger.exception("Error saving data to MongoDB database.")

    def is_valid_data(self, item: Dict[str, Any]) -> bool:
        """Validate the data before saving it to the database."""
        required_fields = ['title', 'link', 'snippet', 'image_urls', 'video_urls', 'source', 'sentiment', 'entities', 'topics']
        return all(field in item for field in required_fields)

class TestDataSaving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a test spider."""
        cls.spider = MH370Spider()

    def test_save_valid_data(self):
        """Test saving valid data to the database."""
        valid_data = [
            {'title': 'Valid Title 1', 'link': 'https://example.com/1', 'snippet': 'Valid Snippet 1', 'image_urls': ['https://example.com/image1.jpg'], 'video_urls': ['https://example.com/video1.mp4'], 'source': 'Test', 'sentiment': 0.5, 'entities': ['entity1', 'entity2'], 'topics': ['topic1', 'topic2']}
        ]
        asyncio.run(self.spider.save_data(valid_data))
        # Check if the data was saved to the database
        saved_data = MH370Data.select().where(MH370Data.title == 'Valid Title 1')
        self.assertEqual(len(saved_data), 1)
        
    def test_save_empty_data(self):
        """Test saving empty data to the database."""
        asyncio.run(self.spider.save_data([]))
        # Check if the data was not saved to the database
        saved_data = MH370Data.select().where(MH370Data.title == 'Valid Title 1')
        self.assertEqual(len(saved_data), 0)

    def test_error_handling(self):
        """Test error handling during data saving."""
        invalid_data = [{'invalid': 'data'}]  # This data is missing required fields
        with self.assertRaises(Exception):
            asyncio.run(self.spider.save_data(invalid_data))

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        MH370Data.delete().where(MH370Data.title == 'Valid Title 1').execute()

# Unit tests for the migration process
class TestMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a test spider."""
        cls.spider = MH370Spider()

    def test_migration(self):
        """Test the migration process."""
        old_schema_data = [
            {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'image_urls': ['Image 1'],
             'video_urls': ['Video 1'], 'source': 'Test'},
            {'title': 'Title 2', 'link': 'Link 2', 'snippet': 'Snippet 2', 'image_urls': ['Image 2'],
             'video_urls': ['Video 2'], 'source': 'Test'}
        ]
        # Save the old schema data to the old database
        for item in old_schema_data:
            MH370Data.create(**item)
        # Migrate the data
        self.spider.migrate_data()
        # Check if the data was migrated to the new database
        new_schema_data = MH370Data.select()
        self.assertEqual(len(new_schema_data), len(old_schema_data))
        for old_item, new_item in zip(old_schema_data, new_schema_data):
            self.assertEqual(old_item['title'], new_item.title)
            self.assertEqual(old_item['link'], new_item.link)
            self.assertEqual(old_item['snippet'], new_item.snippet)
            self.assertEqual(old_item['image_urls'], new_item.image_urls.split(','))
            self.assertEqual(old_item['video_urls'], new_item.video_urls.split(','))
            self.assertEqual(old_item['source'], new_item.source)

    def test_invalid_data_validation(self):
        """Test case for invalid data validation."""
        invalid_data = {'title': 'Title 1', 'link': 'Link 1', 'source': 'Test'}
        self.assertFalse(self.spider.is_valid_data(invalid_data))

    def test_valid_data_validation(self):
        """Test case for valid data validation."""
        valid_data = {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'source': 'Test'}
        self.assertTrue(self.spider.is_valid_data(valid_data))

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        MH370Data.delete().where(MH370Data.title.in_(['Title 1', 'Title 2'])).execute()

class TestBackup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a test spider."""
        cls.spider = MH370Spider()
        cls.backup_file_path = os.getenv("BACKUP_FILE_PATH", "backup_file.txt")  # Get the backup file path from environment variables

    def test_backup_creation(self):
        """Test the creation of a backup file."""
        self.spider.create_backup(self.backup_file_path)
        # Check if the backup file was created
        self.assertTrue(os.path.exists(self.backup_file_path))

    def test_backup_content(self):
        """Test the content of the backup file."""
        self.spider.create_backup(self.backup_file_path)
        # Check if the backup file contains the correct data
        with open(self.backup_file_path, 'r') as backup_file:
            backup_data = json.load(backup_file)
        database_data = [item for item in MH370Data.select()]
        self.assertEqual(len(backup_data), len(database_data))
        for backup_item, database_item in zip(backup_data, database_data):
            self.assertEqual(backup_item['title'], database_item.title)
            self.assertEqual(backup_item['link'], database_item.link)
            self.assertEqual(backup_item['snippet'], database_item.snippet)
            self.assertEqual(backup_item['image_urls'], database_item.image_urls.split(','))
            self.assertEqual(backup_item['video_urls'], database_item.video_urls.split(','))
            self.assertEqual(backup_item['source'], database_item.source)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if os.path.exists(cls.backup_file_path):
            os.remove(cls.backup_file_path)

class MH370Spider:
    def __init__(self):
        """Initialize the spider."""
        self.api_client = APIClient()

    # Twitter
    @retry(stop=stop_after_attempt(MAX_RETRIES))
    async def fetch_twitter_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Twitter API."""
        try:
            self.logger.info(f"Fetching Twitter data for query: {query}")
            # Initialize Twitter API client
            api = self.init_twitter_api()
            # Fetch data from Twitter API
            tweets = api.search(q=query, tweet_mode='extended', count=100)
            # Process fetched data
            twitter_data = []
            for tweet in tweets:
                # Perform sentiment analysis
                sentiment = TextBlob(tweet.full_text).sentiment.polarity
                # Perform entity recognition
                doc = self.nlp(tweet.full_text)
                entities = [ent.text for ent in doc.ents]
                # Perform topic modeling
                topics = self.model[doc]
                twitter_data.append({
                    'title': tweet.user.name,
                    'link': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                    'snippet': tweet.full_text,
                    'image_urls': [media['media_url_https'] for media in tweet.entities.get('media', [])],
                    'video_urls': [],  # Twitter API does not provide video URLs
                    'source': 'Twitter',
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            return twitter_data
        except tweepy.TweepError as e:
            self.logger.error(f'Error fetching Twitter data: {str(e)}')
            self.logger.exception("Exception occurred during fetching Twitter data.")
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error fetching Twitter data: {str(e)}')
            self.logger.exception("Unexpected exception occurred during fetching Twitter data.")
            raise

    def init_twitter_api(self):
        """Initialize Twitter API client."""
        auth = tweepy.OAuth1UserHandler(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET"), os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
        return tweepy.API(auth, timeout=NETWORK_TIMEOUT)
    
    # Reddit
    @retry(stop=stop_after_attempt(MAX_RETRIES))
    async def fetch_reddit_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Reddit API."""
        try:
            self.logger.info(f"Fetching Reddit data for query: {query}")
            # Initialize Reddit API client
            reddit = self.init_reddit_api()
            # Fetch data from Reddit API
            posts = reddit.subreddit('all').search(query, limit=100)
            # Process fetched data
            reddit_data = []
            for post in posts:
                # Perform sentiment analysis
                sentiment = TextBlob(post.selftext).sentiment.polarity
                # Perform entity recognition
                doc = self.nlp(post.selftext)
                entities = [ent.text for ent in doc.ents]
                # Perform topic modeling
                topics = self.model[doc]
                reddit_data.append({
                    'title': post.title,
                    'link': post.url,
                    'snippet': post.selftext,
                    'image_urls': self.extract_image_urls(post),
                    'video_urls': self.extract_video_urls(post),
                    'source': 'Reddit',
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            return reddit_data
        except praw.exceptions.PRAWException as e:
            self.logger.error(f'Error fetching Reddit data: {str(e)}')
            self.logger.exception("Exception occurred during fetching Reddit data.")
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error fetching Reddit data: {str(e)}')
            self.logger.exception("Unexpected exception occurred during fetching Reddit data.")
            raise

    def init_reddit_api(self):
        """Initialize Reddit API client."""
        return praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"), client_secret=os.getenv("REDDIT_CLIENT_SECRET"), user_agent=os.getenv("REDDIT_USER_AGENT"))

    def extract_image_urls(self, post):
        """Extract image URLs from the post data."""
        return [post.preview['images'][0]['source']['url']] if post.preview else []

    def extract_video_urls(self, post):
        """Extract video URLs from the post data."""
        return [post.media['oembed']['thumbnail_url']] if post.media else []
    
    # Instagram
    @retry(stop=stop_after_attempt(MAX_RETRIES))
    async def fetch_instagram_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from Instagram API."""
        try:
            self.logger.info(f"Fetching Instagram data for query: {query}")
            # Initialize Instagram API client
            instagram = self.init_instagram_api()
            # Fetch data from Instagram API
            posts = instagram.search(query, limit=100)
            # Process fetched data
            instagram_data = []
            for post in posts:
                # Perform sentiment analysis
                sentiment = TextBlob(post.caption).sentiment.polarity
                # Perform entity recognition
                doc = self.nlp(post.caption)
                entities = [ent.text for ent in doc.ents]
                # Perform topic modeling
                topics = self.model[doc]
                instagram_data.append({
                    'title': post.user.username,
                    'link': post.link,
                    'snippet': post.caption,
                    'image_urls': self.extract_image_urls(post),
                    'video_urls': self.extract_video_urls(post),
                    'source': 'Instagram',
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            return instagram_data
        except InstagramAPIError as e:
            self.logger.error(f'Error fetching Instagram data: {str(e)}')
            self.logger.exception("Exception occurred during fetching Instagram data.")
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error fetching Instagram data: {str(e)}')
            self.logger.exception("Unexpected exception occurred during fetching Instagram data.")
            raise

    def init_instagram_api(self):
        """Initialize Instagram API client."""
        return InstagramAPI(os.getenv("INSTAGRAM_USERNAME"), os.getenv("INSTAGRAM_PASSWORD"))

    def extract_image_urls(self, post):
        """Extract image URLs from the post data."""
        return [post.image_url] if post.image_url else []

    def extract_video_urls(self, post):
        """Extract video URLs from the post data."""
        return [post.video_url] if post.video_url else []
    
    # YouTube
    @retry(stop=stop_after_attempt(MAX_RETRIES))
    async def fetch_youtube_data(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from YouTube API."""
        try:
            self.logger.info(f"Fetching YouTube data for query: {query}")
            # Initialize YouTube API client
            youtube_api = self.init_youtube_api()
            # Fetch data from YouTube API
            youtube_videos = await youtube_api.search(query, limit=100)
            # Process fetched data
            youtube_data = []
            for video in youtube_videos:
                # Perform sentiment analysis
                sentiment = TextBlob(video.description).sentiment.polarity
                # Perform entity recognition
                doc = self.nlp(video.description)
                entities = [ent.text for ent in doc.ents]
                # Perform topic modeling
                topics = self.model[doc]
                youtube_data.append({
                    'title': video.title,
                    'link': f"https://www.youtube.com/watch?v={video.id}",
                    'snippet': video.description,
                    'image_urls': [video.thumbnail_url],
                    'video_urls': [f"https://www.youtube.com/watch?v={video.id}"],
                    'source': 'YouTube',
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            return youtube_data
        except YouTubeAPIError as e:
            self.logger.error(f'Error fetching YouTube data: {str(e)}')
            self.logger.exception("Exception occurred during fetching YouTube data.")
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error fetching YouTube data: {str(e)}')
            self.logger.exception("Unexpected exception occurred during fetching YouTube data.")
            raise

    def init_youtube_api(self):
        """Initialize YouTube API client."""
        return YouTubeAPI(os.getenv("YOUTUBE_API_KEY"))
    
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def retry_fetch(self, fetch_function):
        """Retry fetching data with exponential backoff."""
        try:
            return await fetch_function()
        except Exception as e:
            self.logger.error(f'Retry failed: {str(e)}')
            self.logger.exception("Retry failed.")
            raise
        self.logger.error("All retry attempts failed. Unable to fetch data.")
        return []

class TestMH370Spider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up the test data."""
        self.mock_search = patch('mh370_spider.tweepy.API.search').start()
        self.spider = MH370Spider()

    async def test_fetch_twitter_data_success(self):
        """Test fetching Twitter data successfully."""
        # Mock the Twitter API response
        self.mock_search.return_value.items.return_value = [
            Mock(user=Mock(screen_name='user1'), full_text='tweet 1', created_at='2024-03-18 10:00:00'),
            Mock(user=Mock(screen_name='user2'), full_text='tweet 2', created_at='2024-03-18 10:10:00')
        ]
        # Fetch Twitter data
        twitter_data = await self.spider.fetch_twitter_data('MH370')
        # Check if Twitter data is fetched correctly
        self.assertEqual(len(twitter_data), 2)
        self.assertEqual(twitter_data[0]['title'], 'user1')
        self.assertEqual(twitter_data[0]['snippet'], 'tweet 1')
        self.assertEqual(twitter_data[1]['title'], 'user2')
        self.assertEqual(twitter_data[1]['snippet'], 'tweet 2')

    async def test_fetch_twitter_data_empty_response(self):
        """Test fetching Twitter data with empty response."""
        # Mock the Twitter API response
        self.mock_search.return_value.items.return_value = []
        # Fetch Twitter data
        twitter_data = await self.spider.fetch_twitter_data('MH370')
        # Check if Twitter data is fetched correctly
        self.assertEqual(len(twitter_data), 0)

    async def test_fetch_reddit_data_success(self):
        """Test fetching Reddit data successfully."""
        # Mock the Reddit API response
        mock_posts = [Mock(author=Mock(name='author1'), title='Title 1', selftext='Text 1', created_utc=1647588000),
                      Mock(author=Mock(name='author2'), title='Title 2', selftext='Text 2', created_utc=1647588100)]
        mock_reddit_instance = self.mock_reddit.return_value
        mock_reddit_instance.subreddit.return_value.search.return_value = iter(mock_posts)
        # Fetch Reddit data
        reddit_data = await self.spider.fetch_reddit_data('MH370')
        # Check if Reddit data is fetched correctly
        self.assertEqual(len(reddit_data), 2)
        self.assertEqual(reddit_data[0]['title'], 'author1')
        self.assertEqual(reddit_data[0]['snippet'], 'Text 1')
        self.assertEqual(reddit_data[1]['title'], 'author2')
        self.assertEqual(reddit_data[1]['snippet'], 'Text 2')

    async def test_fetch_reddit_data_empty_response(self):
        """Test fetching Reddit data with empty response."""
        # Mock the Reddit API response
        mock_reddit_instance = self.mock_reddit.return_value
        mock_reddit_instance.subreddit.return_value.search.return_value = iter([])
        # Fetch Reddit data
        reddit_data = await self.spider.fetch_reddit_data('MH370')
        # Check if Reddit data is fetched correctly
        self.assertEqual(len(reddit_data), 0)

    async def test_fetch_facebook_data_success(self):
        """Test fetching Facebook data successfully."""
        # Mock the Facebook API response
        self.mock_get_object.return_value = {'data': []}  # Add sample data here if necessary
        # Fetch Facebook data
        facebook_data = await self.spider.fetch_facebook_data('MH370')
        # Check if Facebook data is fetched correctly
        self.assertEqual(len(facebook_data), 0)

    async def test_fetch_instagram_data_success(self):
        """Test fetching Instagram data successfully."""
        # Mock the Instagram API response
        mock_instagram_instance = self.mock_instagram.return_value
        mock_instagram_instance.search.return_value = [
            Mock(user=Mock(username='user1'), caption='Title 1', created_at='2024-03-18 10:00:00'),
            Mock(user=Mock(username='user2'), caption='Title 2', created_at='2024-03-18 10:10:00')
        ]
        # Fetch Instagram data
        instagram_data = await self.spider.fetch_instagram_data('MH370')
        # Check if Instagram data is fetched correctly
        self.assertEqual(len(instagram_data), 2)
        self.assertEqual(instagram_data[0]['title'], 'user1')
        self.assertEqual(instagram_data[0]['snippet'], 'Title 1')
        self.assertEqual(instagram_data[1]['title'], 'user2')
        self.assertEqual(instagram_data[1]['snippet'], 'Title 2')

    async def test_fetch_instagram_data_empty_response(self):
        """Test fetching Instagram data with empty response."""
        # Mock the Instagram API response
        mock_instagram_instance = self.mock_instagram.return_value
        mock_instagram_instance.search.return_value = []
        # Fetch Instagram data
        instagram_data = await self.spider.fetch_instagram_data('MH370')
        # Check if Instagram data is fetched correctly
        self.assertEqual(len(instagram_data), 0)

    async def test_fetch_youtube_data_success(self):
        """Test fetching YouTube data successfully."""
        # Mock the YouTube API response
        mock_youtube_instance = self.mock_youtube.return_value
        mock_youtube_instance.search.return_value = [
            Mock(id='id1', title='Title 1', description='Text 1', thumbnail_url='thumbnail_url1'),
            Mock(id='id2', title='Title 2', description='Text 2', thumbnail_url='thumbnail_url2')
        ]
        # Fetch YouTube data
        youtube_data = await self.spider.fetch_youtube_data('MH370')
        # Check if YouTube data is fetched correctly
        self.assertEqual(len(youtube_data), 2)
        self.assertEqual(youtube_data[0]['title'], 'Title 1')
        self.assertEqual(youtube_data[0]['snippet'], 'Text 1')
        self.assertEqual(youtube_data[1]['title'], 'Title 2')
        self.assertEqual(youtube_data[1]['snippet'], 'Text 2')

    async def test_fetch_youtube_data_empty_response(self):
        """Test fetching YouTube data with empty response."""
        # Mock the YouTube API response
        mock_youtube_instance = self.mock_youtube.return_value
        mock_youtube_instance.search.return_value = []
        # Fetch YouTube data
        youtube_data = await self.spider.fetch_youtube_data('MH370')
        # Check if YouTube data is fetched correctly
        self.assertEqual(len(youtube_data), 0)

    def test_parse_google(self):
        """Test parsing Google search results."""
        # Mock response object
        response = Mock()
        response.text = """
        <html>
            <body>
                <div class="g">
                    <h3 class="r"><a href="https://www.example.com/1">Title 1</a></h3>
                    <div class="s"><div class="st">Snippet 1</div></div>
                </div>
                <div class="g">
                    <h3 class="r"><a href="https://www.example.com/2">Title 2</a></h3>
                    <div class="s"><div class="st">Snippet 2</div></div>
                </div>
            </body>
        </html>
        """
        # Parse Google search results
        google_data = self.spider.parse_google(response)
        # Check if Google data is parsed correctly
        self.assertEqual(len(google_data), 2)
        self.assertEqual(google_data[0]['title'], 'Title 1')
        self.assertEqual(google_data[0]['snippet'], 'Snippet 1')
        self.assertEqual(google_data[1]['title'], 'Title 2')
        self.assertEqual(google_data[1]['snippet'], 'Snippet 2')

    def test_parse_google_empty_response(self):
        """Test parsing Google search results with empty response."""
        # Mock response object
        response = Mock()
        response.text = "<html><body></body></html>"
        # Parse Google search results
        google_data = self.spider.parse_google(response)
        # Check if Google data is parsed correctly
        self.assertEqual(len(google_data), 0)

    def test_save_data_valid(self):
        """Test saving valid data to MongoDB database."""
        # Valid data to save
        data = [
            {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'image_urls': ['Image 1'],
             'video_urls': ['Video 1'], 'source': 'Test'},
            {'title': 'Title 2', 'link': 'Link 2', 'snippet': 'Snippet 2', 'image_urls': ['Image 2'],
             'video_urls': ['Video 2'], 'source': 'Test'}
        ]
        # Save data
        self.spider.save_data(data)
        # Check if data is saved correctly
        saved_data = MH370Data.select()
        self.assertEqual(len(saved_data), len(data))
        for saved_item, data_item in zip(saved_data, data):
            self.assertEqual(saved_item.title, data_item['title'])
            self.assertEqual(saved_item.link, data_item['link'])
            self.assertEqual(saved_item.snippet, data_item['snippet'])
            self.assertEqual(saved_item.image_urls, ','.join(data_item['image_urls']))
            self.assertEqual(saved_item.video_urls, ','.join(data_item['video_urls']))
            self.assertEqual(saved_item.source, data_item['source'])

    @patch('mh370_spider.tk.Tk')
    @patch('mh370_spider.DataMonitorApp')
    @patch('mh370_spider.profile_code')
    @patch('mh370_spider.analyze_profile')
    @patch('mh370_spider.unittest.main')
    def test_main(self, mock_tk, mock_app, mock_profile_code, mock_analyze_profile, mock_unittest_main):
        """Test the main function."""
        # Call the main function
        main()
        # Check if Tk, DataMonitorApp, profile_code, analyze_profile, and unittest.main are called
        mock_tk.assert_called_once()
        mock_app.assert_called_once()
        mock_profile_code.assert_called_once()
        mock_analyze_profile.assert_called_once()
        mock_unittest_main.assert_called_once()

    @patch('mh370_spider.cProfile.Profile')
    def test_profile_code(self, mock_profile):
        """Test the profile_code function."""
        # Call the profile_code function
        profile_code()
        # Check if cProfile.Profile is called
        mock_profile.assert_called_once()

    @patch('mh370_spider.pstats.Stats')
    def test_analyze_profile(self, mock_stats):
        """Test the analyze_profile function."""
        # Call the analyze_profile function
        analyze_profile()
        # Check if pstats.Stats is called with the correct argument
        mock_stats.assert_called_once_with('mh370_profiling')

    @patch('mh370_spider.MH370Spider.fetch_twitter_data')
    @patch('mh370_spider.MH370Spider.fetch_reddit_data')
    @patch('mh370_spider.MH370Spider.fetch_instagram_data')
    @patch('mh370_spider.MH370Spider.fetch_youtube_data')
    async def test_fetch_all_data(self, mock_fetch_twitter_data, mock_fetch_reddit_data, mock_fetch_instagram_data, mock_fetch_youtube_data):
        """Test fetching all data."""
        # Mock the fetch methods to return empty lists
        mock_fetch_twitter_data.return_value = []
        mock_fetch_reddit_data.return_value = []
        mock_fetch_instagram_data.return_value = []
        mock_fetch_youtube_data.return_value = []
        # Initialize MH370Spider and fetch all data
        spider = MH370Spider()
        all_data = await spider.fetch_all_data('MH370')
        # Check if all data is fetched correctly
        self.assertEqual(len(all_data), 0)

    async def test_fetch_all_data_with_data(self):
        """Test fetching all data with actual data."""
        # Mock the fetch methods to return actual data
        self.spider.fetch_twitter_data = Mock(return_value=[{'source': 'Twitter', 'data': 'Data'}])
        self.spider.fetch_reddit_data = Mock(return_value=[{'source': 'Reddit', 'data': 'Data'}])
        self.spider.fetch_instagram_data = Mock(return_value=[{'source': 'Instagram', 'data': 'Data'}])
        self.spider.fetch_youtube_data = Mock(return_value=[{'source': 'YouTube', 'data': 'Data'}])
        # Fetch all data
        all_data = await self.spider.fetch_all_data('MH370')
        # Check if all data is fetched correctly
        self.assertEqual(len(all_data), 4)
        self.assertEqual(all_data[0]['source'], 'Twitter')
        self.assertEqual(all_data[1]['source'], 'Reddit')
        self.assertEqual(all_data[2]['source'], 'Instagram')
        self.assertEqual(all_data[3]['source'], 'YouTube')

    def test_save_data_invalid(self):
        """Test saving invalid data to MongoDB database."""
        # Invalid data to save
        data = [
            {'invalid': 'data'}
        ]
        # Save data
        with self.assertRaises(Exception):
            self.spider.save_data(data)

    def test_migrate_data(self):
        """Test migrating data from the old database to the new database."""
        # Old schema data to migrate
        old_data = [
            {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'image_urls': ['Image 1'],
            'video_urls': ['Video 1'], 'source': 'Test'},
            {'title': 'Title 2', 'link': 'Link 2', 'snippet': 'Snippet 2', 'image_urls': ['Image 2'],
            'video_urls': ['Video 2'], 'source': 'Test'}
        ]
        # Save the old schema data to the old database
        for item in old_data:
            MH370Data.create(**item)
        # Migrate data
        self.spider.migrate_data()
        # Check if data is migrated correctly
        migrated_data = MH370Data.select()
        self.assertEqual(len(migrated_data), len(old_data))
        for migrated_item, old_item in zip(migrated_data, old_data):
            self.assertEqual(migrated_item.title, old_item['title'])
            self.assertEqual(migrated_item.link, old_item['link'])
            self.assertEqual(migrated_item.snippet, old_item['snippet'])
            self.assertEqual(migrated_item.image_urls, ','.join(old_item['image_urls']))
            self.assertEqual(migrated_item.video_urls, ','.join(old_item['video_urls']))
            self.assertEqual(migrated_item.source, old_item['source'])
        
    def tearDown(self):
        """Clean up the test data."""
        patch.stopall()

def profile_code():
    """Run the code with profiling and print the profiling results."""
    # Run the code with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    unittest.main()
    profiler.disable()
    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

def analyze_profile():
    """Analyze the profiling results and print the top 10 functions by cumulative time."""
    # Load the profiling results
    stats = pstats.Stats('mh370_profiling')
    # Remove the directory names from the function names
    stats.strip_dirs()
    # Sort the statistics by the cumulative time spent in the function
    stats.sort_stats('cumulative')
    # Print the statistics for the top 10 functions
    stats.print_stats(10)

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = DataMonitorApp(root)
    root.mainloop()
    profile_code()
    analyze_profile()
    unittest.main()

if __name__ == '__main__':
    with db.connection_context():
        db.create_tables([MH370Data], safe=True)
    main()