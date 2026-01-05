# Quick Start Guide

## Prerequisites

1. **Docker** must be installed and running
2. **Python 3.10+** with pip

## Installation

```bash
# Install dependencies
pip install pyvespa sentence-transformers numpy

# Or use the requirements file
pip install -r layered_ranking_demo/requirements.txt
```

## Running the Demo

```bash
# From the project root directory
python layered_ranking_demo/main.py
```

## What Happens

1. The script creates 5 sample documents about different topics
2. Each document is chunked into multiple pieces
3. Embeddings are generated for each chunk
4. Vespa is deployed in Docker with both ranking approaches
5. The same queries are run with both approaches
6. Results are compared side-by-side

## Expected Output

You'll see:
- Document feeding progress
- Query results for traditional ranking (all chunks)
- Query results for layered ranking (top 3 chunks only)
- Comparison showing the difference

## Stopping the Demo

Press `Ctrl+C` to stop. The Vespa Docker container will continue running.

To stop the container:
```bash
docker ps  # Find the container ID
docker stop <container_id>
```

## Troubleshooting

### "Docker not running"
- Make sure Docker Desktop (or Docker daemon) is running
- On macOS, ensure Docker Desktop is started

### "sentence-transformers model download"
- First run will download the model (~80MB)
- This is normal and only happens once

### "Vespa deployment failed"
- Check Docker has enough resources (4GB+ RAM recommended)
- Try stopping other containers to free resources

### "No results found"
- Wait a few seconds after feeding documents (indexing takes time)
- Increase the sleep time in the code if needed

