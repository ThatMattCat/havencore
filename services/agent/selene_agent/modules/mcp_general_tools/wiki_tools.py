import requests
import json
from typing import Optional

async def query_wikipedia(search_term: str, sentences: int = 7, lang: str = 'en') -> str:
    """
    Query Wikipedia API and return a summary of the search term.
    
    Args:
        search_term: The topic to search for on Wikipedia
        sentences: Number of sentences to return in the summary (1-10)
        lang: Language code for Wikipedia (default: 'en' for English)
    
    Returns:
        A string containing the Wikipedia summary or an error message
    """
    
    # Validate input
    if not search_term or not search_term.strip():
        return "Error: Empty search term provided."
    
    sentences = max(1, min(10, sentences))  # Clamp between 1 and 10
    
    # Wikipedia API endpoint
    base_url = f"https://{lang}.wikipedia.org/api/rest_v1"
    
    try:
        # First, search for the page
        search_url = f"{base_url}/page/summary/{requests.utils.quote(search_term)}"
        
        headers = {
            'User-Agent': 'AI-Assistant/1.0 (https://example.com/contact)'
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        
        if response.status_code == 404:
            # Try searching with the search API if direct page not found
            return await search_wikipedia_fallback(search_term, sentences, lang)
        
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant information
        title = data.get('title', 'Unknown')
        extract = data.get('extract', '')
        
        if not extract:
            return f"No content found for '{search_term}'."
        
        # Format the response for the LLM
        result = f"Wikipedia: {title}\n\n{extract}"
        
        # Add disambiguation note if needed
        if data.get('type') == 'disambiguation':
            result = f"Note: '{search_term}' is a disambiguation page. {result}"
        
        return result
        
    except requests.exceptions.Timeout:
        return f"Error: Wikipedia API request timed out for '{search_term}'."
    except requests.exceptions.RequestException as e:
        return f"Error accessing Wikipedia API: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response from Wikipedia API."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


async def search_wikipedia_fallback(search_term: str, sentences: int = 3, lang: str = 'en') -> str:
    """
    Fallback function using Wikipedia's search API when direct page lookup fails.
    
    Args:
        search_term: The topic to search for
        sentences: Number of sentences for the summary
        lang: Language code
    
    Returns:
        A string containing the Wikipedia summary or an error message
    """
    try:
        # Use the OpenSearch API for searching
        search_api = f"https://{lang}.wikipedia.org/w/api.php"
        
        search_params = {
            'action': 'opensearch',
            'search': search_term,
            'limit': 1,
            'namespace': 0,
            'format': 'json'
        }
        
        headers = {
            'User-Agent': 'AI-Assistant/1.0 (https://example.com/contact)'
        }
        
        response = requests.get(search_api, params=search_params, headers=headers, timeout=5)
        response.raise_for_status()
        
        search_results = response.json()
        
        if len(search_results) > 1 and len(search_results[1]) > 0:
            # Get the first search result
            page_title = search_results[1][0]
            
            # Now get the summary for this page
            summary_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'exsentences': sentences,
                'titles': page_title
            }
            
            response = requests.get(search_api, params=summary_params, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if page_id != '-1':  # -1 means page not found
                    title = page_data.get('title', 'Unknown')
                    extract = page_data.get('extract', '')
                    
                    if extract:
                        return f"Wikipedia: {title}\n\n{extract}"
        
        return f"No Wikipedia results found for '{search_term}'."
        
    except Exception as e:
        return f"Search fallback error: {str(e)}"


async def get_wikipedia_with_context(search_term: str, include_categories: bool = False) -> str:
    """
    Enhanced function that includes additional context for the AI assistant.
    
    Args:
        search_term: The topic to search for
        include_categories: Whether to include category information
    
    Returns:
        A formatted string with Wikipedia content and metadata
    """
    
    # Get the basic summary
    summary = await query_wikipedia(search_term, sentences=5)
    
    if summary.startswith("Error") or summary.startswith("No"):
        return summary
    
    # Optionally add more context
    if include_categories:
        try:
            # This would require additional API calls for categories
            # Simplified for this example
            result = f"{summary}\n\n[Additional context can be retrieved if needed]"
        except:
            result = summary
    else:
        result = summary
    
    return result


# Example usage
if __name__ == "__main__":
    # Test the function
    test_queries = [
        "Quantum Computing",
        "python programming language",
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = query_wikipedia(query)
        print(result)
        print()
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        context_result = get_wikipedia_with_context(query, include_categories=True)
        print(context_result)
        print()