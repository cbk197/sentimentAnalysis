import scrapy

from scrapy.crawler import CrawlerProcess

    



class crawlData(scrapy.Spider):
    name = "brickset_spider"
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/'] 
    content = []
    typeComment = []
    def __init__(self, *argv):
        super()
        for u in argv:
            self.typeComment.append(int(u,10))
           
    
    
    
    def start_requests(self):
        print("============start===========",self.typeComment[0])
        return super().start_requests()

    def parse(self, response):
        quotes = response.xpath("//div[@class='quote']//span[@class='text']/text()").extract()
        self.content.append(quotes)
        print("=============quo=============",quotes)
        return self.content

        
process = CrawlerProcess(settings={
    'FEED_FORMAT': 'json',
    'FEED_URI': 'items.json'
})
m = crawlData('1')
process.crawl(m)
process.start() # the script will block here until the crawling is finished

class getData:
    def getComment(numberComment, content ):
        pass

