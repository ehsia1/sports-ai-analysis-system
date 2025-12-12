# Docker Setup Guide

## Prerequisites

### 1. Install Docker Desktop

**macOS:**
```bash
# Option A: Download from Docker website
# https://www.docker.com/products/docker-desktop/

# Option B: Homebrew
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

**Verify installation:**
```bash
docker --version
docker compose version
```

## Project Docker Files

### 2. Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

# Create data directory
RUN mkdir -p data

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/orchestrate.py", "status"]
```

### 3. docker-compose.yml

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  betting:
    build: .
    container_name: nfl-betting
    volumes:
      # Persist database
      - ./data:/app/data
      # Persist model files
      - ./models:/app/models
      # Config cache
      - betting-cache:/root/.sports_betting
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
      - DATABASE_URL=sqlite:///data/sports_betting.db
      - LOG_LEVEL=INFO
    env_file:
      - .env

  # Scheduled runner (optional)
  scheduler:
    build: .
    container_name: nfl-betting-scheduler
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - betting-cache:/root/.sports_betting
    env_file:
      - .env
    command: python scripts/scheduler.py
    restart: unless-stopped

volumes:
  betting-cache:
```

### 4. .dockerignore

Create `.dockerignore` in project root:

```
venv/
__pycache__/
*.pyc
.git/
.env
*.md
tests/
docs/
.pytest_cache/
.coverage
htmlcov/
```

## Usage

### Build the image

```bash
docker compose build
```

### Run commands

```bash
# Check status
docker compose run --rm betting python scripts/orchestrate.py status

# Pre-game workflow
docker compose run --rm betting python scripts/orchestrate.py pre-game

# Post-game workflow
docker compose run --rm betting python scripts/orchestrate.py post-game

# Generate parlays
docker compose run --rm betting python scripts/orchestrate.py parlay --notify

# Health check
docker compose run --rm betting python scripts/orchestrate.py health
```

### Interactive shell

```bash
docker compose run --rm betting bash
```

## Automated Scheduling (Phase 4b)

### Option A: Simple cron on host

Add to crontab (`crontab -e`):

```cron
# Pre-game: Thursday 6pm, Sunday 11am (before games)
0 18 * * 4 cd /path/to/sports-betting && docker compose run --rm betting python scripts/orchestrate.py pre-game --notify
0 11 * * 0 cd /path/to/sports-betting && docker compose run --rm betting python scripts/orchestrate.py pre-game --notify

# Post-game: Monday/Tuesday 6am (after games complete)
0 6 * * 1,2 cd /path/to/sports-betting && docker compose run --rm betting python scripts/orchestrate.py post-game
```

### Option B: APScheduler in container

Create `scripts/scheduler.py`:

```python
"""Automated scheduler for NFL betting workflows."""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
from loguru import logger

scheduler = BlockingScheduler()

def run_command(cmd: list[str]):
    """Run orchestrate command."""
    full_cmd = ["python", "scripts/orchestrate.py"] + cmd
    logger.info(f"Running: {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
    else:
        logger.info(f"Command completed: {result.stdout[-500:]}")

# Thursday Night Football - pre-game at 6pm ET
@scheduler.scheduled_job(CronTrigger(day_of_week='thu', hour=18, timezone='America/New_York'))
def thursday_pregame():
    run_command(["pre-game", "--notify"])

# Sunday games - pre-game at 11am ET
@scheduler.scheduled_job(CronTrigger(day_of_week='sun', hour=11, timezone='America/New_York'))
def sunday_pregame():
    run_command(["pre-game", "--notify"])

# Monday morning - post-game scoring
@scheduler.scheduled_job(CronTrigger(day_of_week='mon', hour=6, timezone='America/New_York'))
def monday_postgame():
    run_command(["post-game"])

# Daily health check at 9am ET
@scheduler.scheduled_job(CronTrigger(hour=9, timezone='America/New_York'))
def daily_health():
    run_command(["health", "--notify"])

if __name__ == "__main__":
    logger.info("Starting NFL Betting Scheduler")
    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")
    scheduler.start()
```

Add APScheduler to requirements:
```bash
pip install apscheduler
echo "apscheduler>=3.10.0" >> requirements.txt
```

Run scheduler:
```bash
docker compose up -d scheduler
docker compose logs -f scheduler
```

## AWS EC2 Deployment (Phase 4c)

### 1. Launch EC2 instance

- **Instance type**: t3.micro (free tier) or t3.small
- **AMI**: Amazon Linux 2023 or Ubuntu 22.04
- **Storage**: 20GB gp3
- **Security group**: SSH (22) only

### 2. Install Docker on EC2

```bash
# Amazon Linux 2023
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install docker compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

### 3. Deploy code

```bash
# Clone repo (or scp files)
git clone https://github.com/yourusername/sports-betting.git
cd sports-betting

# Create .env
cat > .env << EOF
ODDS_API_KEY=your_key_here
DISCORD_WEBHOOK_URL=your_webhook_here
EOF

# Build and run
docker compose build
docker compose up -d scheduler
```

### 4. CloudWatch Logging (optional)

Add to docker-compose.yml:

```yaml
services:
  betting:
    logging:
      driver: awslogs
      options:
        awslogs-group: nfl-betting
        awslogs-region: us-east-1
        awslogs-stream-prefix: betting
```

## Checklist

- [ ] Install Docker Desktop
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create .dockerignore
- [ ] Test build: `docker compose build`
- [ ] Test status: `docker compose run --rm betting python scripts/orchestrate.py status`
- [ ] Test pre-game workflow
- [ ] (Optional) Create scheduler.py
- [ ] (Optional) Deploy to EC2
