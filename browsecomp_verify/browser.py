# Based in part on Hannibal046/GPT-OSS-BrowseCompPlus-Eval; see THIRD_PARTY_NOTICES.md.
import logging
from aiohttp import ClientSession
from typing import AsyncIterator
import time,json,html
from gpt_oss.tools.simple_browser.page_contents import (
    process_html,
)
from gpt_oss.tools.simple_browser.backend import (
    VIEW_SOURCE_PREFIX,
    BackendError,
    maybe_truncate,
)

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool,maybe_get_function_args,function_the_model_can_call,handle_errors
from gpt_oss.tools.simple_browser.backend import BackendError, maybe_truncate

logger = logging.getLogger(__name__)

class BrowserTool(SimpleBrowserTool):
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return
        try:
            if function_name == "search":
                async for msg in self.search(**function_args):
                    yield msg
            elif function_name == "open":
                async for msg in self.open(**function_args):
                    yield msg
            elif function_name == "find":
                async for msg in self.find(**function_args):
                    yield msg
        except TypeError as e:
            error_text = f"Error: Invalid arguments for function '{function_name}'. Please check the function signature. Details: {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
        except Exception as e:
            error_text = f"An unexpected error occurred while executing function '{function_name}': {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
    
    async def _open_url(self, url: str, direct_url_open: bool):
        """Use the cache, if available."""
        backend = self.backend
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            async with ClientSession() as session:
                page = await backend.fetch(url, session=session)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

class LocalServiceBrowserBackend:
    source = "web"
    
    def __init__(self,base_url):
        self.base_url = base_url
        
    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        t0 = time.time()
        async with session.post(f"{self.base_url}{endpoint}", json=payload) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            return await resp.json()
                
    async def search(
        self, 
        query: str, 
        topn: int = 5,
        session = None,
    ):
        data = await self._post(
            session,
            "/search",
            {"query": query, "topn": topn},
        )
        results = data.get("results",[])
        if not results:
            raise BackendError(f"No results returned for query: '{query}'")
        
        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result['title'],quote=True),
                html.escape(result['url'],quote=True),
                html.escape(result['summary'],quote=True)   
            ))
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary])}
</ul>
</body></html>
"""     
        
        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=query,
            display_urls=True,
            session=session,
        )
        
    async def fetch(self, url:str, session=None):
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        
        data = await self._post(
            session,
            "/get_content",
            {"url": url},
        )
        
        if not data or not data.get("content"):
            raise BackendError(f"No content returned for {url}")
        return process_html(
            html=data.get("content", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )