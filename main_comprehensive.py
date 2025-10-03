# main_curated_playlists.py - Use Known Popular/Curated Hip-Hop Playlists
import os
import itertools
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -------------------- CONFIG --------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep all collabs; set to 2+ if you want only repeated collaborators.
MIN_EDGE_WEIGHT = 1

# Seed for deterministic layout
LAYOUT_SEED = 42

# If True, filter to artists whose Spotify genres include hip hop / rap / trap
FILTER_TO_HIPHOP = True

# How many to show in betweenness bar chart
TOP_N_BETWEENNESS = 20

# -------------------- AUTH --------------------
load_dotenv()  # loads .env if present
SCOPES = ["playlist-read-private", "playlist-read-collaborative"]
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))

# -------------------- HELPERS --------------------
def get_all_playlist_tracks(sp, playlist_id):
    """Fetch *all* items from a playlist (paginated)."""
    items = []
    results = sp.playlist_items(playlist_id, additional_types=["track"], limit=100)
    items.extend(results.get("items", []))
    while results.get("next"):
        results = sp.next(results)
        items.extend(results.get("items", []))
    return items

def is_hiphop(genres_list):
    """Basic hip-hop/rap predicate using artist-level Spotify genres."""
    s = " ".join(genres_list or []).lower()
    return ("hip hop" in s) or ("rap" in s) or ("trap" in s)

# -------------------- KNOWN POPULAR PLAYLIST IDs --------------------
# These are known popular/curated Hip-Hop playlists from Spotify
CURATED_PLAYLISTS = [
    # Spotify Editorial Playlists (most popular)
    {"id": "37i9dQZF1DX0XUsuxWHRQd", "name": "RapCaviar", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX186v583rmzp", "name": "Most Necessary", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX4SBhb3fqCJd", "name": "Rap UK", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DXb8wplbC2YhV", "name": "100 Greatest Hip-Hop Songs", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX1lVhptIry0d", "name": "Hot Country", "type": "Spotify Editorial"},  # This might not be hip-hop
    {"id": "37i9dQZF1DXcBWIGoYBM5M", "name": "Today's Top Hits", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DXcF6B6QPhFDv", "name": "Rock This", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX4dyzvuaRJ0n", "name": "mint", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX4sWSpwq3LiO", "name": "Chill Hits", "type": "Spotify Editorial"},
    {"id": "37i9dQZF1DX0XUsuxWHRQd", "name": "RapCaviar", "type": "Spotify Editorial"},  # Duplicate, will be deduplicated
]

# Alternative approach: Search for playlists with high follower counts
def find_high_follower_playlists():
    """Find playlists by searching and sorting by followers."""
    
    print("🔍 Searching for high-follower Hip-Hop playlists...")
    
    # Search terms that typically return popular playlists
    search_terms = [
        "hip hop",
        "rap",
        "trap",
        "hip hop hits",
        "rap hits",
        "hip hop 2024",
        "rap 2024",
        "hip hop 2025",
        "rap 2025"
    ]
    
    all_playlists = []
    
    for term in search_terms:
        try:
            print(f"   Searching for: '{term}'")
            results = sp.search(q=term, type='playlist', limit=20, market='US')
            playlists = results.get('playlists', {}).get('items', [])
            
            for playlist in playlists:
                if playlist and playlist.get('id'):
                    all_playlists.append({
                        'id': playlist.get('id'),
                        'name': playlist.get('name'),
                        'followers': playlist.get('followers', {}).get('total', 0),
                        'owner': playlist.get('owner', {}).get('display_name', 'Unknown'),
                        'description': playlist.get('description', ''),
                        'tracks': playlist.get('tracks', {}).get('total', 0),
                        'search_term': term
                    })
        except Exception as e:
            print(f"   Error searching for '{term}': {e}")
    
    # Remove duplicates and sort by followers
    unique_playlists = {}
    for playlist in all_playlists:
        pid = playlist['id']
        if pid not in unique_playlists or playlist['followers'] > unique_playlists[pid]['followers']:
            unique_playlists[pid] = playlist
    
    # Sort by followers (most popular first)
    sorted_playlists = sorted(unique_playlists.values(), key=lambda x: x['followers'], reverse=True)
    
    return sorted_playlists

# -------------------- COLLECT TRACKS AND ARTISTS --------------------
all_rows = []
all_artists = set()  # Track all unique artists
seen_track_ids = set()

print("🎵 Using curated and popular Hip-Hop playlists...")

# First, try the known curated playlists
playlist_ids = []
playlist_info = []

print("\n1️⃣ Trying known Spotify Editorial playlists...")
for playlist in CURATED_PLAYLISTS:
    try:
        # Test if playlist exists and is accessible
        test_result = sp.playlist(playlist['id'])
        if test_result:
            playlist_ids.append(playlist['id'])
            playlist_info.append({
                'id': playlist['id'],
                'name': playlist['name'],
                'type': playlist['type'],
                'followers': test_result.get('followers', {}).get('total', 0),
                'tracks': test_result.get('tracks', {}).get('total', 0)
            })
            print(f"   ✅ {playlist['name']} - {test_result.get('followers', {}).get('total', 0):,} followers")
    except Exception as e:
        print(f"   ❌ {playlist['name']} - Not accessible: {e}")

# If we don't have enough playlists, search for more
if len(playlist_ids) < 5:
    print(f"\n2️⃣ Only found {len(playlist_ids)} accessible playlists, searching for more...")
    high_follower_playlists = find_high_follower_playlists()
    
    # Add top playlists from search
    for playlist in high_follower_playlists[:10]:  # Top 10 from search
        if playlist['id'] not in playlist_ids:
            playlist_ids.append(playlist['id'])
            playlist_info.append({
                'id': playlist['id'],
                'name': playlist['name'],
                'type': 'Search Result',
                'followers': playlist['followers'],
                'tracks': playlist['tracks']
            })

print(f"\n📊 Using {len(playlist_ids)} playlists for analysis:")
print("=" * 80)

for i, info in enumerate(playlist_info, 1):
    print(f"{i:2d}. {info['name']}")
    print(f"    ID: {info['id']}")
    print(f"    Type: {info['type']}")
    print(f"    Followers: {info['followers']:,}")
    print(f"    Tracks: {info['tracks']}")
    print()

# Process each playlist
print(f"Processing {len(playlist_ids)} playlists...")
for pid in tqdm(playlist_ids):
    try:
        print(f"Processing playlist: {pid}")
        items = get_all_playlist_tracks(sp, pid)
        print(f"Found {len(items)} tracks in playlist")
        
        for it in items:
            t = it.get("track") or {}
            track_id = t.get("id")
            if not track_id or track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)

            artists = t.get("artists", []) or []
            if not artists:
                continue
            
            # Add all artists to our set (including solo artists)
            for artist in artists:
                artist_id = artist.get("id")
                artist_name = artist.get("name")
                if artist_id and artist_name:
                    all_artists.add((artist_id, artist_name))

            # Keep track of collaborations (2+ artists)
            if len(artists) >= 2:
                artist_ids = [a.get("id") for a in artists if a.get("id")]
                artist_names = [a.get("name") for a in artists if a.get("name")]

                all_rows.append({
                    "track_id": track_id,
                    "track_name": t.get("name"),
                    "artist_ids": artist_ids,
                    "artist_names": artist_names,
                })
                
    except spotipy.SpotifyException as e:
        print(f"Error processing playlist {pid}: {e}")
        continue

print(f"Total unique artists found: {len(all_artists)}")
print(f"Collaboration tracks found: {len(all_rows)}")

# Save all tracks (including solo tracks)
df_tracks = pd.DataFrame(all_rows)
df_tracks.to_csv(os.path.join(OUTPUT_DIR, "tracks_raw_curated.csv"), index=False)
print(f"Saved raw tracks: {len(df_tracks)} rows")

# -------------------- BUILD COLLAB EDGES --------------------
edge_weights = defaultdict(int)
for _, row in df_tracks.iterrows():
    ids = row["artist_ids"]
    # Connect all pairs of credited artists on the same track
    for a, b in itertools.combinations(sorted(set(ids)), 2):
        edge_weights[(a, b)] += 1

# Collect unique artists to enrich with names/genres
artist_ids = sorted(set([aid for pair in edge_weights.keys() for aid in pair]))
# Also include all artists we found (even solo ones)
all_artist_ids = [aid for aid, _ in all_artists]
all_artist_ids = sorted(set(all_artist_ids))

id_to_name = {}
id_to_genres = {}

print("Fetching artist metadata (names/genres)...")
# Process artists in batches
for i in tqdm(range(0, len(all_artist_ids), 50)):
    batch = all_artist_ids[i:i+50]
    arts = sp.artists(batch).get("artists", [])
    for a in arts:
        if not a:
            continue
        aid = a.get("id")
        id_to_name[aid] = a.get("name", aid)
        id_to_genres[aid] = a.get("genres", [])

# -------------------- BUILD GRAPH --------------------
G = nx.Graph()

# Add all artists as nodes (including solo artists)
for aid in all_artist_ids:
    G.add_node(aid, name=id_to_name.get(aid, aid), genres=id_to_genres.get(aid, []))

# Add collaboration edges
for (a, b), w in edge_weights.items():
    if w >= MIN_EDGE_WEIGHT:
        G.add_edge(a, b, weight=w)

# Optional genre filter to keep Hip-Hop/Rap subset
if FILTER_TO_HIPHOP:
    keep_nodes = [n for n, d in G.nodes(data=True) if is_hiphop(d.get("genres"))]
    H = G.subgraph(keep_nodes).copy()
else:
    H = G.copy()

print(f"Graph built. Nodes: {H.number_of_nodes()}  Edges: {H.number_of_edges()}")

# -------------------- CENTRALITY --------------------
# Betweenness as the "connector" metric
betweenness = nx.betweenness_centrality(H, weight=None, normalized=True)
degree = dict(H.degree())
weighted_degree = dict(H.degree(weight="weight"))

df_cent = pd.DataFrame({
    "artist_id": list(H.nodes()),
    "artist_name": [H.nodes[n].get("name", n) for n in H.nodes()],
    "betweenness": [betweenness.get(n, 0.0) for n in H.nodes()],
    "degree": [degree.get(n, 0) for n in H.nodes()],
    "weighted_degree": [weighted_degree.get(n, 0) for n in H.nodes()],
    "genres": [", ".join(H.nodes[n].get("genres", [])) for n in H.nodes()]
}).sort_values("betweenness", ascending=False)

df_cent.to_csv(os.path.join(OUTPUT_DIR, "artist_centrality_curated.csv"), index=False)
pd.DataFrame(
    [{"source": a, "target": b, "weight": w["weight"]} for a, b, w in H.edges(data=True)]
).to_csv(os.path.join(OUTPUT_DIR, "edges_curated.csv"), index=False)

print("\nTop 10 by Betweenness:")
print(df_cent.head(10)[["artist_name", "betweenness", "degree", "weighted_degree"]].to_string(index=False))

# -------------------- SIMPLE NETWORK PLOT FUNCTION --------------------
def plot_hiphop_network(G, title="Hip-Hop Collaboration Network"):
    """Simple network plotting function similar to the NBA trade network example"""
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", 
            edge_color="gray", font_size=8)
    plt.title(title)
    plt.show()

# -------------------- FIGURE 1: SIMPLE NETWORK PLOT --------------------
print("Creating simple network visualization...")
plot_hiphop_network(H, "Curated Hip-Hop Collaboration Network")

# -------------------- FIGURE 2: NETWORK NODES WEIGHTED --------------------
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(H, seed=LAYOUT_SEED, k=None)
sizes = []
for n in H.nodes():
    sizes.append(200 + 30 * weighted_degree.get(n, 0))

nx.draw_networkx_edges(H, pos, edge_color="lightgray", width=0.8)
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="lightblue", 
                       linewidths=0.5, edgecolors="gray")

# Label only top-k by weighted degree to keep it readable
label_k = 30
top_by_wdeg = set(df_cent.sort_values("weighted_degree", ascending=False).head(label_k)["artist_id"])
labels = {n: H.nodes[n].get("name", n) for n in H.nodes() if n in top_by_wdeg}
nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

plt.title("Hip-Hop/Rap Collaboration Network — Node Size = Weighted Degree (Curated)")
plt.axis("off")
network_png = os.path.join(OUTPUT_DIR, "network_nodes_weighted_curated.png")
plt.tight_layout()
plt.savefig(network_png, dpi=200)
plt.close()
print(f"Saved: {network_png}")

# -------------------- FIGURE 3: BETWENNESS BAR CHART --------------------
top = df_cent.head(TOP_N_BETWEENNESS)
plt.figure(figsize=(12, 6))
bars = plt.bar(top["artist_name"], top["betweenness"])
plt.title("Betweenness Centrality — Top Connectors in Hip-Hop/Rap (Curated)")
plt.xlabel("Artist")
plt.ylabel("Betweenness Centrality")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
bc_png = os.path.join(OUTPUT_DIR, "betweenness_top20_curated.png")
plt.savefig(bc_png, dpi=200)
plt.close()
print(f"Saved: {bc_png}")

# -------------------- FIGURE 4: DEGREE DISTRIBUTION --------------------
plt.figure(figsize=(10, 6))
degrees = [d for n, d in H.degree()]
plt.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Degree Distribution - Hip-Hop Collaboration Network (Curated)")
plt.xlabel("Number of Collaborations")
plt.ylabel("Number of Artists")
plt.grid(True, alpha=0.3)
plt.tight_layout()
dd_png = os.path.join(OUTPUT_DIR, "degree_distribution_curated.png")
plt.savefig(dd_png, dpi=200)
plt.close()
print(f"Saved: {dd_png}")

# -------------------- FIGURE 5: WEIGHTED DEGREE TOP 20 --------------------
top_weighted = df_cent.nlargest(20, 'weighted_degree')
plt.figure(figsize=(12, 6))
bars = plt.bar(top_weighted["artist_name"], top_weighted["weighted_degree"])
plt.title("Top 20 Artists by Weighted Degree (Total Collaborations) - Curated")
plt.xlabel("Artist")
plt.ylabel("Total Collaborations")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
wd_png = os.path.join(OUTPUT_DIR, "weighted_degree_top20_curated.png")
plt.savefig(wd_png, dpi=200)
plt.close()
print(f"Saved: {wd_png}")

# -------------------- GEPHI EXPORT --------------------
print("Exporting network for Gephi...")

# Create a copy of the graph with genres as strings for GraphML compatibility
H_export = H.copy()
for node in H_export.nodes():
    if 'genres' in H_export.nodes[node]:
        # Convert list to comma-separated string
        genres = H_export.nodes[node]['genres']
        if isinstance(genres, list):
            H_export.nodes[node]['genres'] = ', '.join(genres)
    
    # Add artist name as a separate attribute for Gephi
    artist_name = H_export.nodes[node].get('name', node)
    H_export.nodes[node]['label'] = artist_name
    H_export.nodes[node]['artist_name'] = artist_name

# Export to GraphML format for Gephi
nx.write_graphml(H_export, os.path.join(OUTPUT_DIR, "hiphop_network_curated.graphml"))
print(f"Saved: {os.path.join(OUTPUT_DIR, 'hiphop_network_curated.graphml')}")

# -------------------- SUMMARY STATISTICS --------------------
print(f"\n=== NETWORK SUMMARY ===")
print(f"Total Artists: {H.number_of_nodes()}")
print(f"Total Collaborations: {H.number_of_edges()}")

# Calculate basic stats
degrees = dict(H.degree())
avg_degree = sum(degrees.values()) / len(degrees)
print(f"Average Degree: {avg_degree:.2f}")
print(f"Network Density: {nx.density(H):.4f}")

# Find most connected artists
degree_centrality = nx.degree_centrality(H)
top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"\nTop 10 Most Connected Artists:")
for artist_id, centrality in top_connected:
    artist_name = H.nodes[artist_id]['name']
    degree = degrees[artist_id]
    print(f"{artist_name}: {degree} connections (centrality: {centrality:.4f})")

print("\nDone! The network plot should be displayed above.")
print("\nTo use in Gephi:")
print("1. Open Gephi")
print("2. File > Open > Select 'hiphop_network_curated.graphml'")
print("3. The network will load with artist names and collaboration weights")