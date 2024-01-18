from .utils import ToolABC
import time
import httpx
from duckduckgo_search import DDGS
from newspaper import Article as NsArticle, Config as NsConfig
#from bs4 import BeautifulSoup

class Web(ToolABC):
    def __init__(self, tool_manager: object):
        self.tool_manager = tool_manager

    def search(self, keywords: str, *, options: dict={}, proxy: str=None, type: int=1, no_sleep: bool=False):
        results = []
        '''
        results = {
            "origin": [],
            "for_agent": [],
        }
        '''
        search_kwargs = {}
        if proxy:
            search_kwargs["proxies"] = proxy
        if "max_results" not in options:
            options.update({ "max_results": 5 })
        with DDGS(**search_kwargs) as ddgs:
            if type == 1:
                for index, result in enumerate(ddgs.text(
                    str(keywords),
                    **options
                )):
                    print(result)
                    results.append({
                        "title": result["title"],
                        "brief": result["body"],
                        "url": result["href"],
                    })
                    '''
                    results["origin"].append(result)
                    results["for_agent"].append({
                        "index": index,
                        "title": result["title"],
                        "body": result["body"],
                    })
                    '''
            elif type == 2:
                for index, result in enumerate(ddgs.news(
                    str(keywords),
                    **options
                )):
                    results.append({
                        "title": result["title"],
                        "brief": result["body"],
                        "url": result["url"],
                        "source": result["source"],
                        "date": result["date"],
                    })
                    '''
                    results["origin"].append(result)
                    results["for_agent"].append({
                        "index": index,
                        "title": result["title"],
                        "body": result["body"],
                        "date": result["date"],
                    })
                    '''
        return results

    def search_definition(self, item_name: str, *, options: dict={}, proxy: str=None):
        return self.search(f'whatis { item_name }', options=options, proxy=proxy)

    def search_news(self, keywords: str, *, options: dict={}, proxy: str=None):
        if "timelimit" not in options:
            options.update({ "timelimit": "w" })
        if "max_results" not in options:
            options.update({ "max_results": 10 })
        return self.search(keywords, options=options, proxy=proxy, type=2)

    def browse(self, url: str, *, retry_times:int = 0, proxy: str=None):
        ns_config = NsConfig()
        if proxy:
            if proxy.startswith("http:"):
                ns_config.proxies = { "http": proxy }
            elif proxy.startswith("https:"):
                ns_config.proxies = { "https": proxy }
        ns_config.request_timeout = 20
        ns_config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ns_config.browser_accept = "*/*"
        ns_config.fetch_images = False
        ns_config.memoize_articles = False
        ns_article = NsArticle(url, config = ns_config)
        ns_article.download()
        ns_article.parse()
        result = ns_article.text
        return result
        '''
        if len(result) < 200:
            request_params = {}
            if proxy:
                request_params.update({ "proxies": proxy })
            response = httpx.get(url, **request_params)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                result = re.sub(r"\n{3,}", "\n\n", soup.get_text())
                return result
            else:
                if retry_times < 3:
                    return self.browse(url, retry_times = retry_times + 1)
                else:
                    return "Can not open the web page."
        else:
            return result
        '''

    def export(self):
        return {
            "search": {
                "desc": "search {keywords}.",
                "args": {
                    "keywords": ("String", "[*Required]keywords to search, seperate keywords by ' '."),
                    "options": {
                        "timelimit": ("String | Null", "'d': last day; 'w': this week; 'm': this month;")
                    },
                },
                "func": self.search,
                "require_proxy": True,
            },
            "search_info": {
                "desc": "search information about {keywords}.",
                "args": {
                    "keywords": ("String", "[*Required]keywords to search, seperate keywords by ' '."),
                    "options": {
                        "timelimit": ("String | Null", "'d': last day; 'w': this week; 'm': this month;")
                    },
                },
                "func": self.search,
                "require_proxy": True,
            },
            "search_news": {
                "desc": "search news about {keywords}.",
                "args": {
                    "keywords": ("String", "[*Required]keywords to search, seperate keywords by ' '."),
                    "options": {
                        "timelimit": ("String | Null", "'d': last day; 'w': this week; 'm': this month;")
                    },
                },
                "func": self.search_news,
                "require_proxy": True,
            },
            "search_definition": {
                "desc": "search definition about {item_name}.",
                "args": {
                    "item_name": ("String", "[*Required]item name to find the definition."),
                },
                "func": self.search_definition,
                "require_proxy": True,
            },
            "browse": {
                "desc": "Browse web page by url.",
                "args": {
                    "url": ("String", "[*Required]url to browse.")
                },
                "func": self.browse,
                "require_proxy": True,
            }
        }

def export():
    return ("Web", Web)