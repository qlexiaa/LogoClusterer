"""
Logo Scraping Pipeline
----------------------
This script scrapes logos from domains, deduplicates them, clusters them, and visualizes the results.
"""

### =================== ###
###   COMMON IMPORTS    ###
### =================== ###
import os
import re
import ssl
import time
import json
import random
import shutil
import concurrent.futures
from io import BytesIO
from collections import defaultdict

import pandas as pd
import urllib.request, urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image, ImageChops
import imagehash

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

### =================== ###
###   CONFIGURATION     ###
### =================== ###
# Directories
SCRAPING_OUTPUT_DIR = "logos"          # For scraped logos
DEDUPED_DIR = "deduped_logos_final"            # For deduplicated logos (one per domain)
UNIQUE_LOGOS_DIR = "unique_logos_final"      # Final cleaned logos after deduplication
DUPLICATE_VIS_DIR = "duplicates_visualization"  # Collages of duplicate groups
CLUSTER_OUTPUT_IMAGE = "all_clusters.png"       # Visualization of clusters

# Parquet file with domains (adjust path as needed)
PARQUET_PATH = "logos.snappy.parquet"

# Clustering parameters
IMAGE_SIZE = (224, 224)          # Input size for clustering embeddings
DISTANCE_THRESHOLD = 0.15        # For hierarchical clustering (cosine distance = 1 - similarity)
LINKAGE = "complete"             # Clustering linkage method
THUMB_SIZE = (128, 128)          # Thumbnail size for visualization
LOGOS_PER_ROW = 25
CLUSTER_MARGIN = 20

# Deduplication threshold for perceptual hash
DUPLICATE_THRESHOLD = 10

### =================== ###
###  STEP 1: SCRAPING   ###
### =================== ###

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def get_root_domain(domain):
    """Extract the root domain (e.g., 'example' from 'example.com')"""
    import tldextract
    extracted = tldextract.extract(domain)
    return extracted.domain

def group_domains_by_root(domains):
    """Group domains by their root name (different TLDs become one group)"""
    domain_groups = {}
    for domain in domains:
        root = get_root_domain(domain)
        domain_groups.setdefault(root, []).append(domain)
    return domain_groups

def download_image(url, headers, context=None, timeout=10):
    """Download an image from a URL"""
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        # print(f"Failed to download image from {url}: {e}")
        return None

def is_valid_image(image_data):
    """Check if image_data is a valid image"""
    if not image_data:
        return False
    try:
        Image.open(BytesIO(image_data))
        return True
    except Exception:
        return False

def preprocess_logo(image_data, size=(128, 128)):
    """Resize and convert logo image to PNG bytes"""
    try:
        img = Image.open(BytesIO(image_data))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255))
            background.paste(img, (0, 0), img)
            img = background.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        output = BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_logo_from_html(soup, base_url, headers, context):
    """Extract candidate logo URLs from HTML using several methods"""
    logo_candidates = []
    # Method 1: favicon links
    favicon_links = soup.find_all("link", rel=["shortcut icon", "icon", "apple-touch-icon", "apple-touch-icon-precomposed"])
    for link in favicon_links:
        href = link.get("href")
        if href:
            if not (href.startswith("http://") or href.startswith("https://")):
                href = urllib.parse.urljoin(base_url, href)
            logo_candidates.append({"url": href, "source": "favicon", "priority": 3})
    default_favicon = urllib.parse.urljoin(base_url, "/favicon.ico")
    logo_candidates.append({"url": default_favicon, "source": "default_favicon", "priority": 4})
    # Method 2: <img> tags with "logo" in class or alt.
    img_tags = soup.find_all("img")
    for img in img_tags:
        classes = img.get("class", [])
        alt = img.get("alt", "")
        src = img.get("src")
        if src and ("logo" in " ".join(classes).lower() or "logo" in alt.lower()):
            if not (src.startswith("http://") or src.startswith("https://")):
                src = urllib.parse.urljoin(base_url, src)
            logo_candidates.insert(0, {"url": src, "source": "img_tag", "priority": 1})
    logo_candidates.sort(key=lambda x: x["priority"])
    for candidate in logo_candidates:
        try:
            image_data = download_image(candidate["url"], headers, context)
            if is_valid_image(image_data):
                processed_data = preprocess_logo(image_data)
                if processed_data:
                    return processed_data, candidate["source"]
        except Exception as e:
            # print(f"Failed candidate {candidate['url']}: {e}")
            continue
    return None, None

def get_logos_from_domain(domain):
    """Attempt to get a logo from a domain using various methods"""
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92.0.4515.107 Safari/537.36",
        ]
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        urls_to_try = [
            f"https://{domain}",
            f"https://www.{domain}",
            f"http://{domain}",
            f"http://www.{domain}",
        ]
        for base_url in urls_to_try:
            try:
                time.sleep(random.uniform(0.1, 0.5))
                request = urllib.request.Request(base_url, headers=headers)
                page = urllib.request.urlopen(request, context=context, timeout=15)
                soup = BeautifulSoup(page, "html.parser")
                logo_data, source = extract_logo_from_html(soup, base_url, headers, context)
                if logo_data:
                    print(f"Found logo for {domain} from {source}")
                    return {
                        "domain": domain,
                        "root_domain": get_root_domain(domain),
                        "logo_data": logo_data,
                        "source": source,
                    }
            except Exception as e:
                # print(f"Failed to process {base_url}: {e}")
                continue
        print(f"No logo found for {domain}")
        return {"domain": domain, "root_domain": get_root_domain(domain), "logo_data": None, "source": None}
    except Exception as e:
        print(f"Error processing {domain}: {e}")
        return {"domain": domain, "root_domain": get_root_domain(domain), "logo_data": None, "source": None}

def save_logo(root_domain, logo_data, output_dir=SCRAPING_OUTPUT_DIR):
    """Save a logo to disk (one per root domain)"""
    if not logo_data:
        return False
    filename = re.sub(r"[^\w\-\.]", "_", root_domain) + ".png"
    path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(logo_data)
    return True

def get_already_processed_domains(output_dir=SCRAPING_OUTPUT_DIR):
    """Return set of domains that already have logos saved."""
    if not os.path.exists(output_dir):
        return set()
    processed = set()
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            domain = filename[:-4]
            domain = re.sub(r"_", ".", domain)
            processed.add(domain)
    return processed

def process_domain_group(domains):
    """Process a group of domains (same root) and return the best logo result."""
    results = []
    for domain in domains:
        result = get_logos_from_domain(domain)
        if result["logo_data"]:
            results.append(result)
    if results:
        return results[0]
    return {"domain": domains[0], "root_domain": get_root_domain(domains[0]), "logo_data": None, "source": None}

def run_scraping():
    """Main scraping function"""
    os.makedirs(SCRAPING_OUTPUT_DIR, exist_ok=True)
    df = read_parquet(PARQUET_PATH)
    print(f"Total domains in dataset: {len(df)}")
    already_processed = get_already_processed_domains()
    print(f"Already processed: {len(already_processed)} domains")
    domains_to_process = [d for d in df["domain"].tolist() if d not in already_processed]
    print(f"Remaining domains: {len(domains_to_process)}")
    if not domains_to_process:
        print("All domains processed.")
        return

    domain_groups = group_domains_by_root(domains_to_process)
    print(f"Grouped into {len(domain_groups)} root domains")
    groups_to_process = []
    for root, group in domain_groups.items():
        sorted_domains = sorted(group, key=len)
        groups_to_process.append(sorted_domains[:min(2, len(sorted_domains))])
    max_workers = min(16, os.cpu_count() * 2)
    print(f"Using {max_workers} workers")
    results = {}
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_group = {executor.submit(process_domain_group, group): group[0] for group in groups_to_process}
        for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(groups_to_process)):
            primary_domain = future_to_group[future]
            try:
                result = future.result()
                root = result["root_domain"]
                if result["logo_data"]:
                    if root not in results:
                        if save_logo(root, result["logo_data"]):
                            results[root] = result
            except Exception as e:
                print(f"Error processing {primary_domain}: {e}")
    duration = time.time() - start_time
    print(f"Scraping completed in {duration:.2f} seconds")
    print(f"Saved {len(results)} logos in '{SCRAPING_OUTPUT_DIR}'")
    return results

### =================== ###
###  STEP 2: DEDUPLICATION  ###
### =================== ###
def deduplicate_logos():
    """Deduplicate logos using perceptual hashing with a threshold of DUPLICATE_THRESHOLD.
       For each group of similar files, copy only one representative to the unique logos directory.
    """
    os.makedirs(UNIQUE_LOGOS_DIR, exist_ok=True)
    os.makedirs(DUPLICATE_VIS_DIR, exist_ok=True)

    seen_hashes = []
    hash_to_files = defaultdict(list)

    def get_logo_phash(path):
        try:
            img = Image.open(path)
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.paste(img, (0, 0), img)
                img = bg.convert("RGB")
            img = img.resize((128, 128), Image.LANCZOS).convert("L")
            return imagehash.phash(img)
        except Exception as e:
            print(f"Failed to hash {path}: {e}")
            return None

    # Process each file in the source directory.
    for filename in tqdm(os.listdir(SCRAPING_OUTPUT_DIR)):
        if not filename.endswith(".png"):
            continue
        src_path = os.path.join(SCRAPING_OUTPUT_DIR, filename)
        hash_val = get_logo_phash(src_path)
        if hash_val is None:
            continue
        duplicate_found = False
        for existing_hash in seen_hashes:
            # If the Hamming distance is within the threshold, consider it a duplicate.
            if abs(hash_val - existing_hash) <= DUPLICATE_THRESHOLD:
                hash_to_files[str(existing_hash)].append(filename)
                duplicate_found = True
                break
        if not duplicate_found:
            seen_hashes.append(hash_val)
            hash_to_files[str(hash_val)].append(filename)
            # Copy this unique representative to the unique logos directory.
            shutil.copy(src_path, os.path.join(UNIQUE_LOGOS_DIR, filename))

    print(f"\nFinished deduplication! Unique logos saved in '{UNIQUE_LOGOS_DIR}'")
    print(f"Total unique logos: {len(seen_hashes)}")

    # Visualize duplicate groups (where more than one file is similar).
    def create_collage(image_files, output_path, thumb_size=(128, 128), logos_per_row=5):
        images = []
        for file in image_files:
            path = os.path.join(SCRAPING_OUTPUT_DIR, file)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(thumb_size)
                images.append(img)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        if not images:
            return
        n = len(images)
        rows = (n + logos_per_row - 1) // logos_per_row
        collage_width = logos_per_row * thumb_size[0]
        collage_height = rows * thumb_size[1]
        collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))
        for i, img in enumerate(images):
            x = (i % logos_per_row) * thumb_size[0]
            y = (i // logos_per_row) * thumb_size[1]
            collage.paste(img, (x, y))
        collage.save(output_path)

    for hash_str, files in hash_to_files.items():
        if len(files) > 1:
            output_vis_path = os.path.join(DUPLICATE_VIS_DIR, f"group_{hash_str}.png")
            create_collage(files, output_vis_path)
            print(f"Created collage for group {hash_str} with {len(files)} images.")

### =================== ###
###  STEP 3: CLUSTERING  ###
### =================== ###
def load_images_for_clustering(image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder, exist_ok=True)
        print(f"Created directory {image_folder} since it did not exist.")
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")])
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            images.append(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return images, image_paths

def extract_embeddings(images, model_name="clip-ViT-B-32"):
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_name)
    print("Encoding images...")
    embeddings = model.encode(images, convert_to_numpy=True)
    return embeddings

def cluster_by_hierarchical(embeddings, distance_threshold=DISTANCE_THRESHOLD, linkage=LINKAGE):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(embeddings_norm)
    return labels, embeddings_norm

def merge_singletons(clusters, embeddings_norm):
    non_singleton_clusters = []
    singletons = []
    for cluster in clusters:
        if len(cluster) == 1:
            singletons.append(cluster[0])
        else:
            non_singleton_clusters.append(cluster)
    if len(non_singleton_clusters) == 0:
        return clusters
    centroids = []
    for cluster in non_singleton_clusters:
        cluster_embeds = embeddings_norm[cluster]
        centroid = np.mean(cluster_embeds, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid /= centroid_norm
        centroids.append(centroid)
    for idx in singletons:
        embed = embeddings_norm[idx]
        similarities = [np.dot(embed, centroid) for centroid in centroids]
        best_cluster_index = np.argmax(similarities)
        non_singleton_clusters[best_cluster_index].append(idx)
    merged_clusters = [sorted(cluster) for cluster in non_singleton_clusters]
    return merged_clusters

def visualize_all_clusters(clusters, image_paths, output_path, thumb_size, logos_per_row, margin):
    cluster_images = []
    for cluster in clusters:
        num_imgs = len(cluster)
        if num_imgs == 0:
            continue
        rows = (num_imgs + logos_per_row - 1) // logos_per_row
        grid_width = logos_per_row * thumb_size[0]
        grid_height = rows * thumb_size[1]
        grid_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
        for idx, img_idx in enumerate(cluster):
            try:
                img = Image.open(image_paths[img_idx]).convert("RGB")
                img = img.resize(thumb_size)
                x = (idx % logos_per_row) * thumb_size[0]
                y = (idx // logos_per_row) * thumb_size[1]
                grid_img.paste(img, (x, y))
            except Exception as e:
                print(f"Error loading {image_paths[img_idx]}: {e}")
        cluster_images.append(grid_img)
    if not cluster_images:
        print("No clusters to display.")
        return
    total_width = max(img.width for img in cluster_images)
    total_height = sum(img.height for img in cluster_images) + margin * (len(cluster_images) - 1)
    final_img = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for grid in cluster_images:
        final_img.paste(grid, (0, y_offset))
        y_offset += grid.height + margin
    final_img.save(output_path)
    print(f"Saved all clusters visualization to {output_path}")

def run_clustering():
    images, image_paths = load_images_for_clustering(UNIQUE_LOGOS_DIR)
    if not images:
        print("No valid images found for clustering.")
        return
    embeddings = extract_embeddings(images)
    labels, embeddings_norm = cluster_by_hierarchical(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
    clusters_dict = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(label, []).append(idx)
    clusters = list(clusters_dict.values())
    print("Clusters before merging singletons:")
    for idx, cluster in enumerate(clusters):
        cluster_files = [os.path.basename(image_paths[i]) for i in cluster]
        print(f"Cluster {idx+1}: {cluster_files}")
    clusters = merge_singletons(clusters, embeddings_norm)
    print("Clusters after merging singletons:")
    for idx, cluster in enumerate(clusters):
        cluster_files = [os.path.basename(image_paths[i]) for i in cluster]
        print(f"Cluster {idx+1}: {cluster_files}")
    visualize_all_clusters(clusters, image_paths, CLUSTER_OUTPUT_IMAGE, THUMB_SIZE, LOGOS_PER_ROW, CLUSTER_MARGIN)

### =================== ###
###       MAIN          ###
### =================== ###
if __name__ == "__main__":
    # Step 1: Scrape logos and save them to SCRAPING_OUTPUT_DIR
    run_scraping()
    
    # Step 2: Deduplicate logos
    deduplicate_logos()
    
    # Step 3: Cluster the unique logos
    run_clustering()

