import logging
import tweepy
import praw
import asyncio
import yaml
import requests
import os
from textblob import TextBlob  # For sentiment analysis
import spacy  # For entity recognition
import gensim  # For topic modeling
import unittest
from unittest.mock import patch, Mock
import cProfile
import pstats
import time
import psutil
from playhouse.sqlite_ext import SqliteExtDatabase
from peewee import Model, SqliteDatabase, CharField, TextField
from requests.exceptions import Timeout, RequestException
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import aiohttp
from instagram_private_api import Client
from googleapiclient import discovery
import tkinter as tk
from tkinter import ttk
import shutil
import schedule

# Configure logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set timeouts and buffer sizes for network operations
NETWORK_TIMEOUT = 10  # seconds
BUFFER_SIZE = 8192  # bytes

# Initialize SQLite database with optimizations
db = SqliteDatabase('mh370_optimized.db', pragmas=(
    ('journal_mode', 'wal'),  # Write-Ahead Logging for better concurrency
    ('cache_size', -1024 * 64)  # Set cache size to 64MB for better performance
))

class OptimizedMH370Data(Model):
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

# Implement data migration
def migrate_data():
    try:
        # Retrieve data from existing MH370Data model
        old_data = list(MH370Data.select())
        print(f"Found {len(old_data)} items in MH370Data.")
        # Migrate data to optimized schema
        with db.atomic():
            for item in old_data:
                OptimizedMH370Data.create(
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
        print("Data migration completed.")
    except Exception as e:
        logger.error(f'Error migrating data: {e}')
        logger.exception("Exception occurred during data migration.")

# Function to initiate data migration process
def initiate_migration(old_schema_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        # Perform migration to the new schema
        with db.atomic():
            for item in old_schema_data:
                OptimizedMH370Data.create(
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

# Function to check if data exists in the new schema as expected
def check_new_schema_data() -> bool:
    try:
        # Check if there is any data in the new schema
        return OptimizedMH370Data.select().exists()
    except Exception as e:
        logger.error(f"Error while checking new schema data: {e}")
        return False

def is_valid_data(data):
    # Perform validation checks here
    if 'title' in data and 'link' in data and 'snippet' in data and 'source' in data:
        return True
    else:
        return False
    
# Implement database backup mechanism
def backup_database():
    try:
        # Specify the path for the backup file
        backup_path = 'mh370_optimized_backup.db'
        # Create a backup by copying the original database file
        shutil.copyfile('mh370_optimized.db', backup_path)
        logger.info("Database backup created successfully.")
    except Exception as e:
        logger.error(f'Error backing up database: {e}')
        logger.exception("Exception occurred during database backup.")

# Function to load configuration from YAML file
def get_config():
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error("Config file not found.")
        logger.exception("Exception occurred while loading config file.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f'Error parsing YAML file: {e}')
        logger.exception("Exception occurred while parsing config file.")
        return {}

def schedule_backup():
    # Schedule backup to run daily at midnight
    schedule.every().day.at("00:00").do(backup_database)

    # Run the scheduler indefinitely
    while True:
        schedule.run_pending()
        time.sleep(1)

# Call the schedule_backup function to start scheduling backups
schedule_backup()

# Create tables if they don't exist
db.connect()
db.create_tables([OptimizedMH370Data])

class DataMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Monitor")
        self.tree = ttk.Treeview(self.root, columns=("Title", "Source", "Sentiment"))
        self.tree.heading("#0", text="ID")
        self.tree.heading("Title", text="Title")
        self.tree.heading("Source", text="Source")
        self.tree.heading("Sentiment", text="Sentiment")
        self.tree.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        refresh_button = ttk.Button(self.root, text="Refresh Data", command=self.refresh_data)
        refresh_button.pack(pady=5)
        add_button = ttk.Button(self.root, text="Add Data", command=self.add_data)
        add_button.pack(pady=5)
        delete_button = ttk.Button(self.root, text="Delete Selected", command=self.delete_selected)
        delete_button.pack(pady=5)
        self.data = [
            {"id": 1, "title": "Example Title 1", "source": "Twitter", "sentiment": "Positive"},
            {"id": 2, "title": "Example Title 2", "source": "Reddit", "sentiment": "Neutral"},
            {"id": 3, "title": "Example Title 3", "source": "Google", "sentiment": "Negative"},
            {"id": 4, "title": "Example Title 3", "source": "YouTube", "sentiment": "Negative"},
            {"id": 5, "title": "Example Title 3", "source": "Facebook", "sentiment": "Negative"},
        ]
        self.populate_treeview()
    
    def populate_treeview(self):
        self.tree.delete(*self.tree.get_children())
        for item in self.data:
            self.tree.insert("", "end", text=item["id"], values=(item["title"], item["source"], item["sentiment"]))
    
    def refresh_data(self):
        import random
        for item in self.data:
            item["sentiment"] = random.choice(["Positive", "Neutral", "Negative"])
        self.populate_treeview()
    
    def add_data(self):
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
        selection = self.tree.selection()
        if selection:
            item_id = self.tree.item(selection[0])["text"]
            for item in self.data:
                if item["id"] == int(item_id):
                    self.data.remove(item)
                    break
            self.populate_treeview()

# Example usage:
config = get_config()
twitter_api_key = config.get('TWITTER_API_KEY', 'your_actual_twitter_api_key')
reddit_client_id = config.get('REDDIT_CLIENT_ID', 'your_actual_reddit_client_id')
reddit_client_secret = config.get('REDDIT_CLIENT_SECRET', 'your_actual_reddit_client_secret')
reddit_user_agent = config.get('REDDIT_USER_AGENT', 'your_actual_reddit_user_agent')
instagram_access_token = config.get('INSTAGRAM_ACCESS_TOKEN', 'your_actual_instagram_access_token')
youtube_api_key = config.get('YOUTUBE_API_KEY', 'your_actual_youtube_api_key')

# Schedule data migration to be executed when needed
migrate_data()

# Class for making API requests
class APIClient:

    def __init__(self):
        # Initialize API clients for different platforms
        self.twitter_api = self.get_twitter_api()
        self.reddit_api = self.get_reddit_api()
        # Add Instagram and YouTube API clients
        self.instagram_api = self.get_instagram_api()
        self.youtube_api = self.get_youtube_api()

    def get_twitter_api(self):
        # Get authenticated Twitter API client.
        auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN,
                                        TWITTER_ACCESS_TOKEN_SECRET)
        return tweepy.API(auth, timeout=NETWORK_TIMEOUT)

    def get_reddit_api(self):
        # Get authenticated Reddit API client.
        return praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)

    def get_instagram_api(self):
        # Implement Instagram API client initialization
        return InstagramAPI()

    def get_youtube_api(self):
        # Implement YouTube API client initialization
        return YouTubeAPI()

# Class for scraping MH370-related data
class MH370Spider:

    def __init__(self):
        # Initialize the spider.
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

    # Google
    async def parse(self, response):
        try:
            google_data = await self.parse_google(response)
            await self.save_data(google_data)
            logger.info("Parsed Google search results and saved data to database.")
            tasks = [
                self.fetch_twitter_data('MH370'),
                self.fetch_reddit_data('MH370'),
                self.fetch_instagram_data('MH370'),  # Add Instagram data fetching task
                self.fetch_youtube_data('MH370')     # Add YouTube data fetching task
            ]
            results = await asyncio.gather(*tasks)
            for result in results:
                await self.save_data(result)
            logger.info("Fetched data from Twitter, Reddit, Instagram, and YouTube APIs and saved to database.")
        except Exception as e:
            logger.error(f'Error in parsing or fetching data: {str(e)}')
            logger.exception("Exception occurred during parsing or fetching data.")

    async def parse_google(self, response):
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
                    'source': 'Google',
                    'sentiment': sentiment,
                    'entities': entities,
                    'topics': topics
                })
            except Exception as e:
                # Log the error and continue to the next result
                logging.error(f"Error parsing Google search result: {e}")
                continue

        logging.debug("Parsed Google search results.")
        return google_data
    
    async def save_data(self, data: List[Dict[str, Any]]):
        try:
            if data:
                with db.atomic():
                    for item in data:
                        if self.is_valid_data(item):
                            OptimizedMH370Data.create(
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
                logger.info("Saved data to SQLite database.")
            else:
                logger.warning("No data to save.")
        except Exception as e:
            logger.error(f'Error saving data to SQLite database: {str(e)}')
            logger.exception("Error saving data to SQLite database.")

class TestDataSaving(unittest.TestCase):
    def test_save_valid_data(self):
        # TODO: Write a test to save valid data to the database
        # Example:
        valid_data = [
            {'title': 'Valid Title 1', 'link': 'https://example.com/1', 'snippet': 'Valid Snippet 1', 'image_urls': ['https://example.com/image1.jpg'], 'video_urls': ['https://example.com/video1.mp4'], 'source': 'Test', 'sentiment': 0.5, 'entities': ['entity1', 'entity2'], 'topics': ['topic1', 'topic2']},
            {'title': 'Valid Title 2', 'link': 'https://example.com/2', 'snippet': 'Valid Snippet 2', 'image_urls': ['https://example.com/image2.jpg'], 'video_urls': ['https://example.com/video2.mp4'], 'source': 'Test', 'sentiment': -0.3, 'entities': ['entity3', 'entity4'], 'topics': ['topic3', 'topic4']}
        ]
        for data in valid_data:
            self.assertTrue(self.is_valid_data(data))
        
        # Additional test cases can be added

    def test_save_empty_data(self):
        # TODO: Write a test to handle saving empty data
        empty_data = []
        self.assertEqual(len(empty_data), 0)
        # Additional test cases can be added

    def test_error_handling(self):
        # TODO: Write a test to ensure proper error handling for database operations
        # Example:
        try:
            # Simulate an error during database operation
            raise Exception("Simulated database error")
        except Exception as e:
            self.assertIsInstance(e, Exception)

# Unit tests for the migration process
class TestMigration(unittest.TestCase):
    def test_migration(self):
        # Sample data in the old schema
        old_schema_data = [
            {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'image_urls': ['Image 1'],
             'video_urls': ['Video 1'], 'source': 'Test'},
            {'title': 'Title 2', 'link': 'Link 2', 'snippet': 'Snippet 2', 'image_urls': ['Image 2'],
             'video_urls': ['Video 2'], 'source': 'Test'}
        ]
        # Initiate the migration process
        migration_result = initiate_migration(old_schema_data)
        # Check if data exists in the new schema as expected
        self.assertTrue(check_new_schema_data())

    def test_invalid_data_validation(self):
        # Test case for invalid data validation
        invalid_data = {'title': 'Title 1', 'link': 'Link 1', 'source': 'Test'}
        self.assertFalse(is_valid_data(invalid_data))

    def test_valid_data_validation(self):
        # Test case for valid data validation
        valid_data = {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'source': 'Test'}
        self.assertTrue(is_valid_data(valid_data))
        
class TestBackup(unittest.TestCase):
    def test_backup_creation(self):
        # TODO: Simulate the creation of a backup file
        backup_file_path = "backup_file.txt"  # Define the path for the backup file
        with open(backup_file_path, "w") as backup_file:
            backup_file.write("Sample backup data")
        # TODO: Verify that the backup file exists in the specified location
        self.assertTrue(os.path.exists(backup_file_path))

    # Twitter
    async def fetch_twitter_data(self, query: str) -> List[Dict[str, Any]]:
        async def fetch_twitter_data_with_retry():
            try:
                logger.info(f"Fetching Twitter data for query: {query}")
                # Authenticate with Twitter API
                auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
                api = tweepy.API(auth, timeout=NETWORK_TIMEOUT)
                # Fetch data from Twitter API
                tweets = api.search(q=query, tweet_mode='extended', count=100)
                # Process fetched data
                twitter_data = []
                for tweet in tweets:
                    if self.is_valid_twitter_data(tweet):
                        twitter_data.append({
                            'user': tweet.user.screen_name,
                            'text': tweet.full_text,
                            'created_at': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'Twitter'
                        })
                logger.info("Fetched data from Twitter API.")
                return twitter_data
            except (tweepy.TweepError, Timeout, RequestException) as e:
                # Handle errors
                logger.error(f'Error fetching data from Twitter API: {str(e)}')
                logger.exception("Error fetching data from Twitter API.")
                raise
        return await self.retry_fetch(fetch_twitter_data_with_retry)

    # Reddit
    async def fetch_reddit_data(self, query: str) -> List[Dict[str, Any]]:
        async def fetch_reddit_data_with_retry():
            try:
                logger.info(f"Fetching Reddit data for query: {query}")
                # Authenticate with Reddit API
                reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
                # Fetch data from Reddit API
                subreddit = reddit.subreddit('all')
                reddit_posts = subreddit.search(query, limit=100)
                # Process fetched data
                reddit_data = []
                for post in reddit_posts:
                    if self.is_valid_reddit_data(post):
                        reddit_data.append({
                            'user': post.author.name,
                            'title': post.title,
                            'text': post.selftext,
                            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc)),
                            'source': 'Reddit'
                        })
                logger.info("Fetched data from Reddit API.")
                return reddit_data
            except (praw.exceptions.PRAWException, Timeout, RequestException) as e:
                logger.error(f'Error fetching data from Reddit API: {str(e)}')
                logger.exception("Error fetching data from Reddit API.")
                raise
        return await self.retry_fetch(fetch_reddit_data_with_retry)

    # Instagram
    async def fetch_instagram_data(self, query: str) -> List[Dict[str, Any]]:
        async def fetch_instagram_data_with_retry():
            try:
                logger.info(f"Fetching Instagram data for query: {query}")
                # Authenticate with Instagram API
                instagram_api = self.api_client.get_instagram_api()
                # Fetch data from Instagram API
                instagram_posts = await instagram_api.search(query, limit=100)
                # Process fetched data
                instagram_data = []
                for post in instagram_posts:
                    if self.is_valid_instagram_data(post):
                        instagram_data.append({
                            'user': post.author.name,
                            'title': post.title,
                            'text': post.selftext,
                            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc)),
                            'source': 'Instagram'
                        })
                logger.info("Fetched data from Instagram API.")
                return instagram_data
            except (aiohttp.ClientError, InstagramApiException) as e:
                logger.error(f'Error fetching data from Instagram API: {str(e)}')
                logger.exception("Error fetching data from Instagram API.")
                raise
        return await self.retry_fetch(fetch_instagram_data_with_retry)

    # YouTube
    async def fetch_youtube_data(self, query: str) -> List[Dict[str, Any]]:
        async def fetch_youtube_data_with_retry():
            try:
                logger.info(f"Fetching YouTube data for query: {query}")
                # Authenticate with YouTube API
                youtube_api = self.api_client.get_youtube_api()
                # Fetch data from YouTube API
                youtube_posts = await youtube_api.search(query, limit=100)
                # Process fetched data
                youtube_data = []
                for post in youtube_posts:
                    if self.is_valid_youtube_data(post):
                        youtube_data.append({
                            'user': post.author.name,
                            'title': post.title,
                            'text': post.selftext,
                            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc)),
                            'source': 'YouTube'
                        })
                logger.info("Fetched data from YouTube API.")
                return youtube_data
            except (aiohttp.ClientError, YouTubeApiException) as e:
                logger.error(f'Error fetching data from YouTube API: {str(e)}')
                logger.exception("Error fetching data from YouTube API.")
                raise
        return await self.retry_fetch(fetch_youtube_data_with_retry)

    async def retry_fetch(self, fetch_function):
        retry_attempts = 3
        retry_delay = 5  # Initial delay in seconds
        for attempt in range(retry_attempts):
            try:
                return await fetch_function()
            except Exception as e:
                logger.error(f'Retry attempt {attempt + 1} failed: {str(e)}')
                logger.exception(f'Retry attempt {attempt + 1} failed.')
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        logger.error("All retry attempts failed. Unable to fetch data.")
        return []
    
class TestMH370Spider(unittest.IsolatedAsyncioTestCase):
    @patch('mh370_spider.tweepy.API.search')
    async def test_fetch_twitter_data_success(self, mock_search):
        """Test fetching Twitter data successfully."""
        # Mock the Twitter API response
        mock_search.return_value.items.return_value = [
            Mock(user=Mock(screen_name='user1'), full_text='tweet 1', created_at='2024-03-18 10:00:00'),
            Mock(user=Mock(screen_name='user2'), full_text='tweet 2', created_at='2024-03-18 10:10:00')
        ]
        # Initialize MH370Spider and fetch Twitter data
        spider = MH370Spider()
        twitter_data = await spider.fetch_twitter_data('MH370')
        # Check if Twitter data is fetched correctly
        self.assertEqual(len(twitter_data), 2)
        self.assertEqual(twitter_data[0]['user'], 'user1')
        self.assertEqual(twitter_data[1]['text'], 'tweet 2')

    @patch('mh370_spider.tweepy.API.search')
    async def test_fetch_twitter_data_empty_response(self, mock_search):
        """Test fetching Twitter data with empty response."""
        # Mock the Twitter API response
        mock_search.return_value.items.return_value = []
        # Initialize MH370Spider and fetch Twitter data
        spider = MH370Spider()
        twitter_data = await spider.fetch_twitter_data('MH370')
        # Check if Twitter data is fetched correctly
        self.assertEqual(len(twitter_data), 0)

    @patch('mh370_spider.praw.Reddit')
    async def test_fetch_reddit_data_success(self, mock_reddit):
        """Test fetching Reddit data successfully."""
        # Mock the Reddit API response
        mock_posts = [Mock(author=Mock(name='author1'), title='Title 1', selftext='Text 1', created_utc=1647588000),
                      Mock(author=Mock(name='author2'), title='Title 2', selftext='Text 2', created_utc=1647588100)]
        mock_reddit_instance = mock_reddit.return_value
        mock_reddit_instance.subreddit.return_value.search.return_value = iter(mock_posts)
        # Initialize MH370Spider and fetch Reddit data
        spider = MH370Spider()
        reddit_data = await spider.fetch_reddit_data('MH370')
        # Check if Reddit data is fetched correctly
        self.assertEqual(len(reddit_data), 2)
        self.assertEqual(reddit_data[0]['user'], 'author1')
        self.assertEqual(reddit_data[1]['title'], 'Title 2')

    @patch('mh370_spider.facebook.GraphAPI.get_object')
    async def test_fetch_facebook_data_success(self, mock_get_object):
        """Test fetching Facebook data successfully."""
        # Mock the Facebook API response
        mock_get_object.return_value = {'data': []}  # Add sample data here if necessary
        # Initialize MH370Spider and fetch Facebook data
        spider = MH370Spider()
        facebook_data = await spider.fetch_facebook_data('MH370')
        # Check if Facebook data is fetched correctly
        self.assertEqual(len(facebook_data), 0)  # Add assertions based on sample data returned

    # Unit tests for Instagram data fetching function
    @patch('mh370_spider.InstagramAPI')
    async def test_fetch_instagram_data_success(self, mock_instagram):
        """Test fetching Instagram data successfully."""
        # Mock the Instagram API response
        mock_instagram_instance = mock_instagram.return_value
        mock_instagram_instance.search.return_value = [
            Mock(author=Mock(username='user1'), title='Title 1', selftext='Text 1', created_utc=1647588000),
            Mock(author=Mock(username='user2'), title='Title 2', selftext='Text 2', created_utc=1647588100)
        ]
        # Initialize MH370Spider and fetch Instagram data
        spider = MH370Spider()
        instagram_data = await spider.fetch_instagram_data('MH370')
        # Check if Instagram data is fetched correctly
        self.assertEqual(len(instagram_data), 2)
        self.assertEqual(instagram_data[0]['user'], 'user1')
        self.assertEqual(instagram_data[1]['title'], 'Title 2')

    @patch('mh370_spider.InstagramAPI')
    async def test_fetch_instagram_data_empty_response(self, mock_instagram):
        """Test fetching Instagram data with empty response."""
        # Mock the Instagram API response
        mock_instagram_instance = mock_instagram.return_value
        mock_instagram_instance.search.return_value = []
        # Initialize MH370Spider and fetch Instagram data
        spider = MH370Spider()
        instagram_data = await spider.fetch_instagram_data('MH370')
        # Check if Instagram data is fetched correctly
        self.assertEqual(len(instagram_data), 0)

    # Unit tests for YouTube data fetching function
    @patch('mh370_spider.YouTubeAPI')
    async def test_fetch_youtube_data_success(self, mock_youtube):
        """Test fetching YouTube data successfully."""
        # Mock the YouTube API response
        mock_youtube_instance = mock_youtube.return_value
        mock_youtube_instance.search.return_value = [
            Mock(author=Mock(username='user1'), title='Title 1', selftext='Text 1', created_utc=1647588000),
            Mock(author=Mock(username='user2'), title='Title 2', selftext='Text 2', created_utc=1647588100)
        ]
        # Initialize MH370Spider and fetch YouTube data
        spider = MH370Spider()
        youtube_data = await spider.fetch_youtube_data('MH370')
        # Check if YouTube data is fetched correctly
        self.assertEqual(len(youtube_data), 2)
        self.assertEqual(youtube_data[0]['user'], 'user1')
        self.assertEqual(youtube_data[1]['title'], 'Title 2')

    @patch('mh370_spider.YouTubeAPI')
    async def test_fetch_youtube_data_empty_response(self, mock_youtube):
        """Test fetching YouTube data with empty response."""
        # Mock the YouTube API response
        mock_youtube_instance = mock_youtube.return_value
        mock_youtube_instance.search.return_value = []
        # Initialize MH370Spider and fetch YouTube data
        spider = MH370Spider()
        youtube_data = await spider.fetch_youtube_data('MH370')
        # Check if YouTube data is fetched correctly
        self.assertEqual(len(youtube_data), 0)

    def test_parse_google(self):
        """Test parsing Google search results."""
        # Mock response object
        response = Mock()
        response.css.return_value = [
            Mock(css=lambda x: 'Title 1', get=lambda: 'Title 1'),
            Mock(css=lambda x: 'https://example.com/link1', get=lambda: 'https://example.com/link1'),
            Mock(css=lambda x: 'Snippet 1', get=lambda: 'Snippet 1'),
            Mock(css=lambda x: ['https://example.com/image1'], getall=lambda: ['https://example.com/image1']),
            Mock(css=lambda x: ['https://www.youtube.com/watch?v=video1'], getall=lambda: ['https://www.youtube.com/watch?v=video1'])
        ]
        # Initialize the spider and call the method to parse Google search results
        spider = MH370Spider()
        google_data = spider.parse_google(response)
        # Assertions
        self.assertIsInstance(google_data, list)
        self.assertTrue(all(isinstance(item, dict) for item in google_data))

    def test_save_data_valid(self):
        """Test saving valid data to SQLite database."""
        # Valid data to save
        data = [
            {'title': 'Title 1', 'link': 'Link 1', 'snippet': 'Snippet 1', 'image_urls': ['Image 1'],
             'video_urls': ['Video 1'], 'source': 'Test'},
            {'title': 'Title 2', 'link': 'Link 2', 'snippet': 'Snippet 2', 'image_urls': ['Image 2'],
             'video_urls': ['Video 2'], 'source': 'Test'}
        ]
        # Call save_data method
        asyncio.run(MH370Spider().save_data(data))
        # Check database for saved data
        saved_data = list(MH370Data.select())
        # Assertions
        self.assertEqual(len(saved_data), 2)
        self.assertEqual(saved_data[0].title, 'Title 1')
        self.assertEqual(saved_data[1].source, 'Test')

def profile_code():
    # Run the code with profiling
    cProfile.run('unittest.main()', 'mh370_profiling')

def analyze_profile():
    # Analyze the profiling results
    stats = pstats.Stats('mh370_profiling')
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()

def main():
    root = tk.Tk()
    app = DataMonitorApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
    # Profile the code
    profile_code()
    # Analyze the profiling results
    analyze_profile()
    unittest.main()