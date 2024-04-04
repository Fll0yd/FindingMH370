import logging
import asyncio
from mh370_spider import MH370Spider

# Configure logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    spider = MH370Spider()
    await spider.fetch_all_data('MH370')

if __name__ == "__main__":
    asyncio.run(main())    