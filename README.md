# LogoClusterer
This project implements an end-to-end pipeline for scraping logos from websites, deduplicating them using perceptual hashing, and clustering the resulting unique logos based on image embeddings. The goal is to obtain a curated set of logo images, grouped by visual similarity, for further analysis or presentation.

## Table of Contents
- [Overview](#overview)
- [Pipeline Steps](#pipeline-steps)
  - [1. Scraping](#1-scraping)
  - [2. Deduplication](#2-deduplication)
  - [3. Clustering](#3-clustering)
- [Possible Improvements](#possible-improvements)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Overview
The pipeline begins by reading a list of domains from a parquet file. For each domain, it attempts to retrieve a logo image by checking several HTML elements (favicons, <img> tags, etc.). Once logos are scraped and saved locally, the deduplication step uses perceptual hashing (pHash) to identify and group near-duplicate images. Only one representative from each duplicate group is retained and copied into a dedicated unique logos directory. Finally, the unique logos are processed by extracting image embeddings using a pre-trained SentenceTransformer model (CLIP-ViT-B-32) and clustering these embeddings with hierarchical clustering (using cosine distance and complete linkage).

---

## Pipeline Steps

### 1. Scraping

The first step in this pipeline involves scraping logos from a list of domain names provided in a Parquet file.

#### Initial Approach: HTML-Based Logo Extraction

I initially attempted to extract logos directly from the HTML of each website. This involved scanning for:

- `<img>` tags with `alt` attributes containing the word `"logo"`.
- Common logo-related class names like `logo`, `site-logo`, `header-logo`, etc.

While this occasionally worked, it proved highly unreliable and inconsistent. In many cases:

- The scraper would extract unrelated elements, such as navigation icons, banners, or decorative images.
- Logos were often embedded in JavaScript-rendered content, making them inaccessible through static scraping.
- Class names varied widely across websites, and some sites did not use meaningful labels.
- The quality, size, and format of extracted images were inconsistent.

#### Final Approach: Favicon-Based Extraction

To make the scraping process more robust and scalable, I switched to extracting **favicons** instead. Favicons are small icon files typically used by browsers and linked in the HTML `<head>` via tags like:

```html
<link rel="icon" href="/favicon.ico">
```

##### Advantages

- Favicons are nearly universally available.
- They are lightweight and quick to download.
- They are referenced consistently in the HTML.
- They can be processed without JavaScript rendering.

##### Trade-offs

The main drawback of this approach is that **favicons do not always accurately represent the brand logo**. This is particularly true for:

- Websites built on CMS platforms like WordPress that use default favicons.
- Sites that use stock or generic icons.

However, in practice, the favicon approach provided far better coverage and consistency than trying to extract full logos from arbitrary HTML.

#### How the Scraper Works

##### 1. Domain Grouping

To avoid downloading the same logo multiple times under different TLDs, domains are grouped by their root domain using `tldextract`. For example:

- `example.com`
- `example.net`
- `www.example.org`

All map to the root domain `example`, and only one is scraped.

##### 2. URL Variants

To improve resilience against broken or incomplete domains, the scraper tries four variations for each domain:

- `https://domain`
- `https://www.domain`
- `http://domain`
- `http://www.domain`

This increases the likelihood of reaching a valid response.

##### 3. User-Agent Rotation

To reduce the chance of getting blocked by websites, the scraper rotates through a list of realistic browser user-agent strings on each request.

##### 4. Candidate Extraction

Using BeautifulSoup, the scraper looks for:

- `<img>` tags with "logo" in the class or alt attributes (high priority).
- `<link>` tags that reference favicons or touch icons (medium priority).
- A fallback to `/favicon.ico` (low priority).

Candidates are sorted by priority and validated in order

Perfect! Here's the **Deduplication** section written in GitHub-friendly Markdown using proper `###` and bullet points, just like you asked — ready for direct use in your `README.md`:

---

### 2. Deduplication

After scraping, the next step in the pipeline is to remove duplicate logos. Since many websites under the same root domain may share identical or near-identical favicons (e.g. white-labeled platforms, WordPress setups, default icons), deduplication is critical before clustering.

#### Why Deduplication Is Necessary

* Multiple domains might point to the same company or CMS.
* Default favicons (e.g., WordPress or hosting icons) are reused widely.
* Without deduplication, clustering would group identical logos multiple times, reducing quality.

##### Approach: Perceptual Hashing (pHash)

To detect duplicates, I used perceptual hashing via the `imagehash` library. A perceptual hash captures the "essence" of an image — allowing similar-looking images to have similar hash values, even if they have minor pixel differences.

* Each logo is resized to 128×128 pixels and converted to grayscale.
* A perceptual hash is computed for each image using `imagehash.phash`.
* The Hamming distance between hashes is used to determine similarity.
* A threshold of `10` was selected — if two hashes differ by 10 bits or fewer, they are considered duplicates.

This method is more flexible than exact file comparison and works even if two logos differ slightly in resolution or background.

#### Step-by-Step Logic

1. **Initialize Structures**
   * A list of seen hashes is kept to track previously encountered logo signatures.
   * A mapping of hash → file names is built for visual inspection later.

2. **Iterate Over Scraped Logos**
   * For each logo in the `logos/` directory:
     * Compute its pHash.
     * Compare it with all previously seen hashes.
     * If it’s similar (Hamming distance ≤ 10), it’s grouped with that hash.
     * Otherwise, it’s considered unique and copied to the `unique_logos_final/` directory.

3. **Save Representative Logos**
   * Only the first (most unique) logo in each group is saved to the final directory.
   * This results in a cleaned-up logo dataset ready for clustering.

4. **Visualize Duplicate Groups**
   * For each group that has more than one logo:
     * A visual collage is generated and saved in the `duplicates_visualization/` folder.
     * This helps manually verify and spot questionable groupings.

#### Limitations & Manual Review

While perceptual hashing does a good job catching most duplicates, it isn’t perfect:

* Some duplicates with slightly more significant design differences may slip through.
* Logos with different aspect ratios or compression artifacts might be misclassified.
* CMS-generated logos may have identical favicons that don’t reflect the brand.

For this reason, **manual review of visualizations** in the `duplicates_visualization/` folder is recommended to fine-tune the final dataset.

#### Output

* Unique logos: saved in `unique_logos_final/`
* Duplicate groups (for review): saved as collages in `duplicates_visualization/`

### 3. Clustering

The final stage in the pipeline involves clustering logos based on visual similarity. This allows the identification of groups of logos that look alike, either because they share stylistic elements, use the same template, or represent related brands.

#### Project Constraint: No Machine Learning

One of the key constraints of this project was to avoid machine learning-based methods for the clustering logic itself. However, to create usable representations of the images, pre-trained visual models (e.g., CLIP) were used only for **embedding generation** — not for classification, training, or prediction. The clustering algorithm itself is deterministic and unsupervised.

#### Step-by-Step Process

##### 1. Embedding Extraction

Although machine learning clustering was not allowed, I needed a numerical representation of each logo to compare visual similarity. To accomplish this, I used the `clip-ViT-B-32` model from the `sentence-transformers` library. This model converts images into 512-dimensional embeddings that capture semantic and visual features.

Each image is:
- Resized to 224×224 pixels
- Converted to RGB format
- Passed through the CLIP model to generate an embedding

The embeddings are then L2-normalized so cosine similarity can be used as a distance metric.

##### 2. Hierarchical Clustering

Agglomerative (hierarchical) clustering was selected because it allows flexible control over clustering behavior without requiring labeled data or a preset number of clusters.

Key settings:
- **Linkage method**: `complete`
  - This requires that all items in a cluster are mutually similar.
  - Prevents "chained" clusters from forming due to outliers.
- **Distance metric**: cosine distance (`1 - cosine similarity`)
- **Distance threshold**: 0.15
  - This roughly corresponds to a cosine similarity of 0.85

The clustering builds a tree of merges and cuts it when the distance between clusters exceeds the threshold.

##### 3. Singleton Merging

After initial clustering, many singleton clusters (i.e., clusters with only one logo) may remain. To reduce fragmentation:
- Singletons are reassigned to their nearest cluster using cosine similarity to cluster centroids.
- This step ensures that every cluster contains at least two logos.

---

#### Why Not Other Clustering Methods?

Several other clustering strategies were considered during development but ultimately rejected:

- **DBSCAN / HDBSCAN**:
  - These density-based methods often resulted in one very large cluster and many logos labeled as noise.
  - They were overly sensitive to small changes in parameters like `min_samples` or `epsilon`, which made them unreliable across different logo datasets.

- **Chain-based threshold clustering** (custom graph traversal):
  - This was the original method, where logos were linked if they exceeded a similarity threshold.
  - However, it created **inconsistent clusters**, as it allowed long transitive chains to connect logos that were not actually similar.

- **Weighted Similarity Fusion (custom multi-metric method)**:
  - I also tried combining several similarity strategies into one system:
    - Feature-based similarity using a `ResNet50` embedding
    - Color histogram correlation
    - Structural similarity (SSIM)
  - The goal was to calculate a blended similarity score for each pair of logos.
  - Despite its complexity, the method failed to produce reliable clusters:
    - It frequently grouped **dissimilar logos** together due to superficial similarities (e.g., color or layout).
    - Results were **noisy**, unstable, and required intensive post-cleaning.
    - The system also relied on deep learning features (ResNet), which blurred the line with machine learning — something I aimed to avoid for the actual clustering phase.

---

The chosen method — **hierarchical clustering with complete linkage** — produced the most consistent, interpretable clusters while respecting the project's constraint to avoid machine learning-based clustering. It offered strong control over cluster tightness and worked well with normalized cosine similarity from CLIP embeddings.

---

#### Output

- Logos are grouped into clusters saved in a single visualization image (`all_clusters.png`).
- Each cluster is displayed as a grid of thumbnails, making it easy to inspect logo similarity manually.

---

## Possible Improvements

While the current pipeline performs well under the given constraints, there are several ways it could be improved:

- **Better logo detection**: Instead of relying on favicon links or `<img>` tags with "logo" in the class/alt text, a more robust image detection system (e.g., heuristic DOM scanning or logo detection models) could improve accuracy.

- **Manual verification interface**: Adding a lightweight GUI or CLI tool for manually confirming or rejecting duplicate clusters would improve quality, especially for borderline cases.

- **Perceptual hash refinement**: Adjusting the perceptual hashing parameters or experimenting with alternative image hashing libraries (e.g., `dhash`, `whash`) may further reduce duplicates.

- **Logo cropping/alignment**: Logos often include extra whitespace, padding, or inconsistent aspect ratios. Preprocessing steps to automatically crop around the content could improve clustering quality.

- **Use of embedding visualization**: Though skipped in this pipeline for simplicity, embedding visualization using PCA or t-SNE could help explore the embedding space and fine-tune clustering thresholds.

- **Cluster splitting**: Large clusters could be automatically re-evaluated and split further using stricter thresholds or hierarchical sub-clustering.

---

## Usage

To run the full pipeline, follow these steps:

1. **Prepare your domain list**:
   - Place your list of domains in a Parquet file named `logos.snappy.parquet`, with at least one column named `domain`.

2. **Run the script**:
   - Execute the main Python file:
     ```bash
     python logo_pipeline.py
     ```
   - The script will:
     - Scrape logo images from the domains
     - Deduplicate them based on perceptual hashing
     - Cluster them based on visual similarity using hierarchical clustering
     - Output the clustered visualization to `all_clusters.png`

3. **Check outputs**:
   - `logos/`: raw scraped logos
   - `unique_logos_final/`: final deduplicated logos used for clustering
   - `duplicates_visualization/`: visual collages of detected duplicates
   - `all_clusters.png`: image showing clusters of similar logos

---

## Conclusion

This project aimed to build a fully functional logo clustering pipeline under the constraint of **not using machine learning for the actual clustering process**. Despite challenges in deduplication and similarity scoring, the final approach using:

- Robust favicon scraping,
- Perceptual hashing for deduplication,
- CLIP-based embeddings for visual understanding,
- And hierarchical clustering for organization,

...achieved interpretable and visually coherent results.

The design favors simplicity, transparency, and control over tuning-heavy or black-box solutions. It also leaves room for optional enhancement through semi-supervised review or more advanced preprocessing.

