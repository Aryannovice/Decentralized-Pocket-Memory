from urllib.parse import urlparse

import httpx

from app.ml.adapters.base import SourceAdapter


class RedditAdapter(SourceAdapter):
    """Adapter for Reddit posts and comments.

    Supports:
    * Individual post URLs (fetches post + all comments)
    * Subreddit URLs (fetches recent top posts)
    * Reddit's public JSON API (no auth required)

    Examples:
    - https://reddit.com/r/python/comments/abc123/title
    - https://reddit.com/r/python
    - https://www.reddit.com/r/learnpython/hot
    """

    source_name = "reddit"

    def read(self, payload: dict) -> str:
        reddit_url = payload.get("url")
        if not reddit_url:
            raise ValueError("Missing url for reddit source.")

        # Normalize URL and append .json to use Reddit's JSON API
        parsed = urlparse(reddit_url)
        if not parsed.netloc:
            reddit_url = f"https://reddit.com{reddit_url}"
            parsed = urlparse(reddit_url)

        host = parsed.netloc.lower()
        if host not in {"reddit.com", "www.reddit.com", "old.reddit.com"}:
            raise ValueError("Reddit source expects a reddit.com URL.")

        # Use Reddit's JSON API
        json_url = reddit_url.rstrip("/") + ".json"
        
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            # Reddit requires a user agent
            headers = {"User-Agent": "DecentralizedPocketMemory/1.0"}
            response = client.get(json_url, headers=headers)
            
            # If 404, try without trailing slash or with limit parameter
            if response.status_code == 404:
                # Try alternate URL formats
                alt_url = reddit_url.split("?")[0].rstrip("/") + ".json?limit=100"
                response = client.get(alt_url, headers=headers)
            
            response.raise_for_status()
            data = response.json()

        # Check if it's a post with comments or a subreddit listing
        if isinstance(data, list) and len(data) >= 2:
            # Post with comments: data[0] is post, data[1] is comments
            return self._parse_post_with_comments(data)
        elif isinstance(data, list) and len(data) == 1:
            # Sometimes Reddit returns just the post without comments list
            return self._parse_post_with_comments(data + [{"data": {"children": []}}])
        elif isinstance(data, dict) and "data" in data:
            # Subreddit listing
            return self._parse_subreddit_listing(data, reddit_url)
        else:
            raise ValueError(f"Unexpected Reddit JSON format: {type(data)}")

    def _parse_post_with_comments(self, data: list) -> str:
        """Parse a Reddit post with its comments."""
        parts = []
        
        # Parse the post itself
        post_data = data[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        author = post_data.get("author", "[deleted]")
        score = post_data.get("score", 0)
        subreddit = post_data.get("subreddit", "")
        
        parts.append(f"Reddit Post from r/{subreddit}")
        parts.append(f"Title: {title}")
        parts.append(f"Author: u/{author} | Score: {score}")
        if selftext:
            parts.append(f"\nPost content:\n{selftext}")
        
        # Parse comments
        if len(data) > 1:
            parts.append("\n\n--- Comments ---\n")
            comments = self._extract_comments(data[1]["data"]["children"])
            parts.extend(comments)
        
        return "\n".join(parts)

    def _parse_subreddit_listing(self, data: dict, url: str) -> str:
        """Parse a subreddit listing (multiple posts)."""
        parts = []
        posts = data["data"]["children"]
        
        subreddit_name = posts[0]["data"].get("subreddit", "unknown") if posts else "unknown"
        parts.append(f"Reddit Posts from r/{subreddit_name}\n")
        
        for post in posts[:25]:  # Limit to top 25 posts
            post_data = post["data"]
            if post_data.get("kind") == "t3" or post.get("kind") == "t3":
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                author = post_data.get("author", "[deleted]")
                score = post_data.get("score", 0)
                num_comments = post_data.get("num_comments", 0)
                
                parts.append(f"\n--- Post ---")
                parts.append(f"Title: {title}")
                parts.append(f"Author: u/{author} | Score: {score} | Comments: {num_comments}")
                if selftext and len(selftext) > 0:
                    # Truncate very long posts
                    text = selftext[:1000] + "..." if len(selftext) > 1000 else selftext
                    parts.append(f"Content: {text}")
        
        return "\n".join(parts)

    def _extract_comments(self, children: list, depth: int = 0, max_depth: int = 3) -> list:
        """Recursively extract comments from Reddit's nested structure."""
        comments = []
        
        if depth > max_depth:
            return comments
        
        for child in children:
            if child.get("kind") == "t1":  # Comment
                comment_data = child["data"]
                author = comment_data.get("author", "[deleted]")
                body = comment_data.get("body", "")
                score = comment_data.get("score", 0)
                
                if body and body != "[deleted]" and body != "[removed]":
                    indent = "  " * depth
                    comments.append(f"{indent}u/{author} (score: {score}):")
                    comments.append(f"{indent}{body}\n")
                    
                    # Recursively get replies
                    replies = comment_data.get("replies")
                    if isinstance(replies, dict) and "data" in replies:
                        reply_children = replies["data"].get("children", [])
                        comments.extend(self._extract_comments(reply_children, depth + 1, max_depth))
        
        return comments
