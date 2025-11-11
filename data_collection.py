"""
Sci-Fi Books Data Collection - Module 4 New
Collects science fiction books from Google Books API
Uses subject: keyword to search for sci-fi books
"""

import requests
import json
import pandas as pd
import time
from typing import List, Dict

class SciFiBookCollector:
    """Collects sci-fi book data from Google Books API"""
    
    def __init__(self, output_dir: str = "data"):
        self.google_books_url = "https://www.googleapis.com/books/v1/volumes"
        self.output_dir = output_dir
        self.books_data = []
        
    def fetch_page(self, start_index: int):
        """
        Fetch a single page of results from Google Books API
        
        Args:
            start_index: Starting index for pagination (0, 40, 80, ...)
        
        Returns:
            JSON response data
        """
        params = {
            'q': 'subject:"Science fiction"',  # Use exact format from reference
            'printType': 'books',
            'langRestrict': 'en',
            'maxResults': 40,
            'startIndex': start_index
        }
        
        response = requests.get(self.google_books_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    
    def fetch_all_books(self, target_count: int = 1000):
        """
        Fetch sci-fi books from Google Books API by paginating through results.
        Collects enough books to find top 100 by ratingsCount.
        
        Args:
            target_count: Target number of books to fetch (500-1000 recommended)
        
        Returns:
            List of book dictionaries
        """
        all_books = []
        start_index = 0
        pages_to_try = 100  # Try more pages to get enough books
        seen_ids = set()  # Track volume IDs to avoid duplicates
        
        print("Fetching sci-fi books from Google Books API...")
        print(f"Target: {target_count} books")
        print('Query: subject:"Science fiction"')
        print("Max results per page: 40")
        
        for page_num in range(pages_to_try):
            try:
                data = self.fetch_page(start_index)
                
                # Check total items available
                if page_num == 0:
                    total_items = data.get('totalItems', 0)
                    print(f"Total items available: {total_items:,}")
                
                page_items = data.get('items', [])
                if not page_items:
                    print(f"\nNo more items found at index {start_index}")
                    break
                
                # Process each book
                for item in page_items:
                    volume_id = item.get('id', '')
                    # Skip duplicates
                    if volume_id in seen_ids:
                        continue
                    seen_ids.add(volume_id)
                    
                    volume_info = item.get('volumeInfo', {})
                    
                    # Extract required fields per reference
                    book_data = {
                        'title': volume_info.get('title', ''),
                        'authors': volume_info.get('authors', []),
                        'categories': volume_info.get('categories', []),
                        'mainCategory': volume_info.get('mainCategory', ''),  # Main category if available
                        'averageRating': volume_info.get('averageRating'),
                        'ratingsCount': volume_info.get('ratingsCount', 0),  # Default to 0 if missing
                        'publishedDate': volume_info.get('publishedDate'),
                        'description': volume_info.get('description', ''),
                        'pageCount': volume_info.get('pageCount'),
                        'publisher': volume_info.get('publisher', ''),
                        'language': volume_info.get('language', ''),
                        'maturityRating': volume_info.get('maturityRating', 'NOT_MATURE'),  # Default to NOT_MATURE if missing
                        'volumeId': volume_id
                    }
                    
                    all_books.append(book_data)
                
                items_in_page = len(page_items)
                print(f"Page {page_num + 1}: Fetched {items_in_page} books (total: {len(all_books)})")
                
                # Update start_index for next page (use actual items returned, not maxResults)
                start_index += items_in_page
                
                # Stop if we've reached our target
                if len(all_books) >= target_count:
                    print(f"\nReached target of {target_count} books")
                    break
                
                # Rate limiting
                time.sleep(0.2)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page at index {start_index}: {e}")
                break
        
        print(f"\nTotal books fetched: {len(all_books)}")
        return all_books
    
    def save_data(self, books: List[Dict], filename: str = "scifi_books.json"):
        """Save books data to JSON file"""
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(books, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filepath}")
    
    def create_dataframe(self, books: List[Dict]) -> pd.DataFrame:
        """Convert books list to DataFrame"""
        # Convert lists to strings for CSV compatibility
        df_data = []
        for book in books:
            row = book.copy()
            # Convert lists to comma-separated strings
            if isinstance(row['authors'], list):
                row['authors'] = ', '.join(row['authors']) if row['authors'] else ''
            if isinstance(row['categories'], list):
                row['categories'] = ', '.join(row['categories']) if row['categories'] else ''
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df
    
    def save_csv(self, books: List[Dict], filename: str = "scifi_books.csv"):
        """Save books data to CSV file"""
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        df = self.create_dataframe(books)
        filepath = f"{self.output_dir}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"CSV saved to {filepath}")
        return df


def main():
    """Main execution function - follows reference pattern exactly"""
    print("=" * 60)
    print("Sci-Fi Books Data Collection - Module 4 New")
    print("Fetching top 200 sci-fi books by ratingsCount")
    print("=" * 60)
    
    collector = SciFiBookCollector(output_dir="data")
    
    # Step 1: Fetch books by paginating through API results
    # Collect 500-1000 books to ensure we have enough with ratings
    print("\nStep 1: Fetching books from Google Books API...")
    books = collector.fetch_all_books(target_count=1000)
    
    if len(books) == 0:
        print("No books fetched! Exiting.")
        return
    
    # Step 2: Convert to DataFrame for processing
    print("\nStep 2: Processing books...")
    df = collector.create_dataframe(books)
    
    # Step 3: Normalize missing ratingsCount to 0 (per reference)
    df['ratingsCount'] = pd.to_numeric(df['ratingsCount'], errors='coerce').fillna(0).astype(int)
    
    print(f"Books with ratingsCount > 0: {(df['ratingsCount'] > 0).sum()}")
    
    # Step 4: De-duplication by title+authors
    print("\nStep 3: De-duplicating by title+authors...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['title', 'authors'], keep='first')
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate(s)")
    
    # Step 5: Sort by ratingsCount (descending) and take top 200
    print("\nStep 4: Sorting by ratingsCount (descending)...")
    top_200 = df.sort_values('ratingsCount', ascending=False).head(200)
    
    print("\nSelected top 200 books by rating count")
    min_rating = top_200['ratingsCount'].min()
    max_rating = top_200['ratingsCount'].max()
    print(f"Rating count range: {min_rating} - {max_rating:,}")
    
    # Convert back to list of dicts for saving
    top_200_list = top_200.to_dict('records')
    # Convert back to original format (lists for authors/categories)
    for book in top_200_list:
        if book['authors'] and isinstance(book['authors'], str):
            book['authors'] = [a.strip() for a in book['authors'].split(',') if a.strip()]
        if book['categories'] and isinstance(book['categories'], str):
            book['categories'] = [c.strip() for c in book['categories'].split(',') if c.strip()]
    
    # Step 6: Save top 200 as JSON and CSV
    print("\nStep 5: Saving results...")
    collector.save_data(top_200_list, "scifi_books.json")
    df_final = collector.save_csv(top_200_list, "scifi_books.csv")
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Data Collection Summary (Top 200 by Rating Count)")
    print("=" * 60)
    print(f"Total books fetched: {len(books)}")
    print(f"Total books in final dataset: {len(top_200_list)}")
    print(f"Books with ratingsCount > 0: {(df_final['ratingsCount'] > 0).sum()}")
    print(f"Books with averageRating: {df_final['averageRating'].notna().sum()}")
    print(f"Books with categories: {df_final['categories'].notna().sum()}")
    print(f"Unique authors: {df_final['authors'].nunique()}")
    print(f"Average rating count: {df_final['ratingsCount'].mean():.0f}")
    print(f"Median rating count: {df_final['ratingsCount'].median():.0f}")
    
    # Show top books
    print("\n" + "=" * 60)
    print("Top 10 Books by Rating Count")
    print("=" * 60)
    top_10 = df_final.nlargest(10, 'ratingsCount')
    print(top_10[['title', 'authors', 'averageRating', 'ratingsCount', 'categories']].to_string(index=False))
    
    # Show category distribution
    print("\n" + "=" * 60)
    print("Category Distribution (Top 200)")
    print("=" * 60)
    # Count categories (they're comma-separated strings now)
    all_categories = []
    for cats in df_final['categories'].dropna():
        if cats:
            all_categories.extend([c.strip() for c in str(cats).split(',')])
    
    from collections import Counter
    category_counts = Counter(all_categories)
    print("\nTop 20 Categories:")
    for cat, count in category_counts.most_common(20):
        print(f"  {cat}: {count}")
    
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

