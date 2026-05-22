# 🚀 Production Deployment Manual: GitHub ➡️ Docker Hub ➡️ AWS EC2

This comprehensive guide walks you through setting up a fully automated, production-ready CI/CD pipeline and server orchestration environment for the **Mental Stress Detection System**. 

By the end of this guide, your application will build automatically on every code push, host its containers in **Docker Hub**, and run securely on a high-availability **AWS EC2** instance complete with database persistence, cached inference, pre-trained machine learning model mounts, and automated updates.

---

## 📐 Production Architecture Overview

```mermaid
graph TD
    Developer[Local Developer] -->|Git Push| GitHub[GitHub Repository]
    
    subgraph GitHub_Actions [GitHub Actions CI/CD]
        CI[Validate: Lint & Test] -->|On Pass| CD[Build & Push Images]
    end
    
    GitHub -->|Triggers Pipeline| CI
    
    subgraph Registry [Docker Hub Registry]
        F_Img[mindsense-frontend:latest]
        B_Img[mindsense-backend:latest]
    end
    
    CD -->|Push Frontend Image| F_Img
    CD -->|Push Backend Image| B_Img
    
    subgraph AWS [AWS EC2 Instance (Ubuntu 24.04 LTS)]
        direction TB
        subgraph Docker_Compose [Docker Compose Stack]
            Nginx[Nginx / React Port 80 / 443]
            Django[Django Backend Port 8000]
            Postgres[(PostgreSQL 15)]
            Redis[(Redis Cache)]
        end
        
        Host_Models[Host: ~/app/backend/saved_models]
        Host_Static[Host: Static & Media Volumes]
    end
    
    F_Img -->|Docker Compose Pull| Nginx
    B_Img -->|Docker Compose Pull| Django
    
    Django -->|Connects| Postgres
    Django -->|Caches| Redis
    Django -->|Mounts Weights| Host_Models
    Nginx -->|Proxies /api| Django
    Nginx -->|Serves Static Files| Host_Static
    
    CD -->|Optional: Remote SSH Deploy Trigger| Docker_Compose
```

---

## 🛠️ Phase 1: Setup Docker Hub & GitHub Secrets

To automate container deployments, your GitHub Action runner must securely authenticate with Docker Hub to push freshly built images.

### 1. Generate a Docker Hub Personal Access Token (PAT)
Using a Personal Access Token (PAT) instead of your primary password is a critical security best practice.
1. Log in to your [Docker Hub Dashboard](https://hub.docker.com/).
2. Click your profile picture in the top-right corner and select **Account Settings**.
3. In the left sidebar, navigate to **Security** and click **Personal Access Tokens**.
4. Click **Create New Token**:
   * **Description**: `github-actions-mindsense-cd`
   * **Access Permissions**: `Read, Write, Delete` (Required to push tags and replace the `latest` image).
5. Click **Generate** and immediately copy the secret key. *(Note: You will never see this token again!)*

### 2. Configure GitHub Repository Secrets
1. Navigate to your GitHub repository webpage.
2. Go to **Settings** (top bar) > **Secrets and variables** (left sidebar) > **Actions**.
3. Under the **Repository secrets** tab, click **New repository secret** to add the following two values:

| Secret Name | Description / Value |
| :--- | :--- |
| **`DOCKERHUB_USERNAME`** | Your exact Docker Hub username (e.g., `janedoe`). |
| **`DOCKERHUB_TOKEN`** | The Personal Access Token (PAT) you copied in the step above. |

Once these are set, the workflow defined in `.github/workflows/cd.yml` will run automatically whenever you push code to `main` or `master` (or tag a release like `v1.0.0`), building and pushing your production images to:
* `yourusername/nlp-project-backend:latest`
* `yourusername/nlp-project-frontend:latest`

---

## ☁️ Phase 2: Provision AWS Infrastructure

We will deploy our containerized Django/React stack on a highly predictable, isolated **Ubuntu Server** using AWS EC2.

### 1. Launch the EC2 Instance
1. Open your [AWS Management Console](https://aws.amazon.com/) and navigate to the **EC2 Dashboard**.
2. Click **Launch Instance** and configure it using the details below:

* **Name**: `mindsense-production`
* **Application and OS Image (AMI)**: Select **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type** (64-bit x86 architecture).
* **Instance Type**: Select **`t3.medium`** (2 vCPUs, 4 GiB RAM).
  > [!IMPORTANT]
  > Because the Django backend loads large pre-trained machine learning model weights (LSTM, SVM, Random Forest) in memory for inference, a `t2.micro` (1 GiB RAM) is highly prone to **Out Of Memory (OOM) crashes** and will terminate the Django process. A `t3.medium` provides the baseline compute needed for model execution.
* **Key Pair**: Select an existing SSH key pair (`.pem` format) or create a new one. Save the `.pem` file safely on your local machine (e.g., `~/.ssh/mindsense-prod.pem`).

### 2. Configure Network Security Group
Create a new Security Group with the following inbound rules to govern network access:

| Type | Protocol | Port Range | Source | Description |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | `22` | `My IP` | Secure access to the server terminal (restricted to your current IP). |
| **HTTP** | TCP | `80` | `Anywhere (0.0.0.0/0)` | Public traffic for the web client. |
| **HTTPS** | TCP | `443` | `Anywhere (0.0.0.0/0)` | Secure public traffic (for SSL setup). |

### 3. Allocate an Elastic IP Address
By default, restarting an EC2 instance changes its public IP address. An Elastic IP keeps your server address static.
1. In the EC2 left sidebar, go to **Network & Security** > **Elastic IPs**.
2. Click **Allocate Elastic IP address** and click **Allocate**.
3. Select the newly created Elastic IP, click **Actions** > **Associate Elastic IP address**.
4. Choose your **`mindsense-production`** instance and click **Associate**.
5. Record your static public IP address (referred to as `<YOUR_EC2_PUBLIC_IP>` below).

---

## 🖥️ Phase 3: Server Setup & Dependencies

Connect to your cloud instance and install the required tools to execute your Docker containers.

### 1. Connect via SSH
From your local terminal, navigate to where your key is stored, adjust its permissions so it's private, and SSH into the instance:
```bash
# Secure the key file permissions (AWS requires this)
chmod 400 ~/.ssh/mindsense-prod.pem

# SSH into Ubuntu
ssh -i ~/.ssh/mindsense-prod.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

### 2. Install Docker and Docker Compose
Run the following script on the EC2 server terminal to update packages and install the modern Docker engine:
```bash
# 1. Update the package registry
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Docker, its CLI, and the systemd system services
sudo apt-get install -y docker.io docker-compose-v2

# 3. Enable Docker to start automatically on system boots
sudo systemctl enable --now docker

# 4. Add the default 'ubuntu' user to the 'docker' system group
# This allows running docker commands without typing 'sudo' every time
sudo usermod -aG docker $USER

# 5. Apply the new group ownership rules to your current session
newgrp docker
```

Verify your installations are functioning correctly:
```bash
docker --version
docker compose version
```

---

## 📦 Phase 4: Prepare Model Storage & Upload Configuration

Your pre-trained machine learning weights (`lstm.pt`, `rf.pkl`, `svm.pkl`, etc.) are gitignored and not built into the Docker images. They must be mounted dynamically from the host server.

### 1. Create the App Workspace on EC2
Inside the EC2 server session, create the target folder structure for your application and pre-trained models:
```bash
mkdir -p ~/app/backend/saved_models
```

### 2. Upload Model Weights (From Local Machine)
Open a **new, local terminal** on your computer. Navigate to your NLP Project root directory and use Secure Copy (`scp`) to upload your locally trained models directly to the EC2 server:
```bash
# Upload all saved model files into the remote EC2 directory
scp -i ~/.ssh/mindsense-prod.pem -r backend/saved_models/* ubuntu@<YOUR_EC2_PUBLIC_IP>:~/app/backend/saved_models/
```

Verify they were transferred correctly. Back on the **EC2 terminal**, list the directory:
```bash
ls ~/app/backend/saved_models/
# Output should contain: lstm.pt, rf.pkl, svm.pkl, lda_model.pkl, tfidf_vectorizer.pkl
```

### 3. Upload the Production Docker Compose Configuration
Transfer `docker-compose.prod.yml` to the production workspace on EC2:
```bash
# Run on your local terminal
scp -i ~/.ssh/mindsense-prod.pem docker-compose.prod.yml ubuntu@<YOUR_EC2_PUBLIC_IP>:~/app/docker-compose.prod.yml
```

---

## 🔒 Phase 5: Configure Production Environments

On the EC2 server, we must construct the environment variables (`.env`) for secrets, database credentials, and API connections.

### 1. Generate the Production `.env` File
On your EC2 terminal, open the terminal editor inside your app folder:
```bash
cd ~/app
nano .env
```

Paste the following template and replace all placeholder values with secure, custom passwords.
```env
# ==========================================
# Docker Hub & Orchestration Configuration
# ==========================================
DOCKERHUB_USERNAME=your_dockerhub_username

# ==========================================
# Django Production Secrets
# ==========================================
# Generate a unique secure key (e.g. using: openssl rand -hex 32)
DJANGO_SECRET_KEY=9d14fca4bcf08c8bde904c6d67e0e854fa16ec48bbbbdecf57db931bc9d2fe1a

# Comma-separated hosts allowed to serve the app
ALLOWED_HOSTS=localhost,127.0.0.1,<YOUR_EC2_PUBLIC_IP>,yourdomain.com

# ==========================================
# Database Configuration (PostgreSQL)
# ==========================================
POSTGRES_DB=stressdb
POSTGRES_USER=stress_admin
POSTGRES_PASSWORD=choose_a_highly_secure_db_password_here

# ==========================================
# Application Configuration & Model Selection
# ==========================================
# Options: 'ensemble', 'lstm', 'rf', 'svm'
ML_DEFAULT_MODEL=ensemble
ML_ALERT_THRESHOLD=4
ML_LLM_ENABLED=false

# Optional: Remote LLM capabilities (uncomment if using models fallback)
# OPENAI_API_KEY=your-openai-api-key
# GEMINI_API_KEY=your-gemini-api-key
```

Save the file by pressing `Ctrl + O`, hitting `Enter`, and exit with `Ctrl + X`.

---

## 🚀 Phase 6: Orchestration & Launch

Let's initialize our complete container stack on the EC2 host.

### 1. Pull the Images from Docker Hub
Download the latest versions of your custom frontend and backend containers:
```bash
cd ~/app
docker compose -f docker-compose.prod.yml pull
```

### 2. Build and Start the Orchestration Services
Launch PostgreSQL, Redis, Django, and Nginx/React running in daemon mode:
```bash
docker compose -f docker-compose.prod.yml up -d
```

### 3. Verify System Health
Check that all four core containers are up and running:
```bash
docker compose -f docker-compose.prod.yml ps
```
Your output should indicate that `mindsense_db_prod`, `mindsense_redis_prod`, `mindsense_backend_prod`, and `mindsense_frontend_prod` are all healthy or running.

### 4. Initialize Django Database & Create Admin User
Your `docker-compose.prod.yml` runs database migrations and collects static files automatically on startup. Next, log in to create an administrator account to access the backend admin panel:
```bash
docker compose -f docker-compose.prod.yml exec backend python manage.py createsuperuser
```
Follow the prompts in the terminal to set an admin username, email, and secure password.

### 5. Access the Platform
Open a web browser on your computer and navigate to:
* **Frontend Web Client**: `http://<YOUR_EC2_PUBLIC_IP>/`
* **Django Admin Panel**: `http://<YOUR_EC2_PUBLIC_IP>/api/admin/`

---

## ⚡ Phase 7: Automating Deployment Updates (True CD)

Instead of manually logging into your EC2 server and running `docker compose pull` every time your code changes, you can automate deployment directly from GitHub Actions by configuring an SSH trigger.

### 1. Add SSH Deployment Credentials to GitHub
Generate a secure deployment key pair (or use your existing EC2 `.pem` key). In your GitHub repository webpage under **Settings** > **Secrets and variables** > **Actions**, add three more repository secrets:

| Secret Name | Description / Value |
| :--- | :--- |
| **`EC2_HOST`** | Your EC2 Static Elastic IP Address (`<YOUR_EC2_PUBLIC_IP>`). |
| **`EC2_USERNAME`** | Set to `ubuntu` (default OS username). |
| **`EC2_SSH_KEY`** | Paste the entire content of your AWS `.pem` private key file. |

### 2. Append a Deploy Job to `.github/workflows/cd.yml`
You can extend the CD pipeline to connect directly to your EC2 instance and restart containers with the newly compiled images. Add a `deploy` job that relies on `build-and-push` completing successfully:

```yaml
  deploy:
    name: Deploy to AWS EC2
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
    
    steps:
      - name: SSH into EC2 and Pull/Restart Images
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          port: 22
          script: |
            cd ~/app
            # Fetch the new images
            docker compose -f docker-compose.prod.yml pull
            # Perform zero-downtime recreation of modified services
            docker compose -f docker-compose.prod.yml up -d --remove-orphans
```

---

## 🔒 Phase 8: Secure Your Site with HTTPS (Let's Encrypt SSL)

To run a secure, encrypted HTTPS production site (`https://yourdomain.com`), configure an SSL certificate using Let's Encrypt and Certbot on your EC2 host.

### 1. Map Your Custom Domain to the Elastic IP
Go to your domain provider (e.g., Namecheap, GoDaddy, Route 53) and add two DNS records:
* **Type A**: `@` ➡️ `<YOUR_EC2_PUBLIC_IP>`
* **Type A**: `www` ➡️ `<YOUR_EC2_PUBLIC_IP>`

### 2. Install Certbot on EC2 Host
Connect to your EC2 terminal and install Certbot to obtain free SSL certificates:
```bash
sudo apt-get install -y certbot python3-certbot-nginx
```

### 3. Temporarily Expose Nginx Port 80 to Certbot
1. Stop your existing docker containers:
   ```bash
   docker compose -f docker-compose.prod.yml down
   ```
2. Request a certificate from Let's Encrypt:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com
   ```
3. Enter your email and agree to terms. Certbot will store the SSL certificates under:
   * **Certificate**: `/etc/letsencrypt/live/yourdomain.com/fullchain.pem`
   * **Private Key**: `/etc/letsencrypt/live/yourdomain.com/privkey.pem`

### 4. Enable Nginx & Compose to Mount SSL Certificates
To allow the frontend Nginx web server to read these SSL certificates, we must map them from the host into the Docker frontend container.

Update the **`frontend`** service block in your `docker-compose.prod.yml` as follows:
```yaml
  frontend:
    image: ${DOCKERHUB_USERNAME}/nlp-project-frontend:latest
    container_name: mindsense_frontend_prod
    restart: unless-stopped
    depends_on:
      - backend
    ports:
      - "80:80"
      - "443:443"  # Expose HTTPS Port
    volumes:
      # Mount Let's Encrypt certificates directly into Nginx
      - /etc/letsencrypt:/etc/letsencrypt:ro
```

Update your **`nginx.conf`** file to handle HTTPS handshake protocols:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect all HTTP requests to secure HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Rebuild/restart your stack:
```bash
docker compose -f docker-compose.prod.yml up -d
```
Your application is now fully encrypted with A+ grade SSL security!

---

## 🛠️ Phase 9: Quick Maintenance Reference

Keep these terminal commands handy on your EC2 host for rapid troubleshooting and maintenance:

* **View real-time engine logs:**
  ```bash
  docker compose -f docker-compose.prod.yml logs -f --tail=100
  ```
* **Check system resource (RAM/CPU) utilization:**
  ```bash
  docker stats
  ```
* **Verify running processes on backend container:**
  ```bash
  docker compose -f docker-compose.prod.yml top
  ```
* **Execute an interactive SQL console against your production DB:**
  ```bash
  docker compose -f docker-compose.prod.yml exec db psql -U stress_admin -d stressdb
  ```
* **Take an on-demand SQL dump/backup of the database:**
  ```bash
  docker compose -f docker-compose.prod.yml exec db pg_dump -U stress_admin stressdb > ~/backup_$(date +%F).sql
  ```
* **Force pull and update the stack manually:**
  ```bash
  docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d
  ```
