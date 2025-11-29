# Deployment Guide for MISAI on Render

This guide walks you through deploying the MISAI application (FastAPI backend + React frontend) as a **single unified service** on Render.

## Architecture

The deployment uses a single web service where:
- **Backend**: FastAPI serves API endpoints
- **Frontend**: Built React app is served as static files by the backend
- **Build Process**: Frontend is built during deployment, then served by FastAPI

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Push your MISAI code to GitHub
3. **API Keys**: Obtain the following API keys:
   - [Google Gemini API Key](https://makersuite.google.com/app/apikey)
   - [Groq API Key](https://console.groq.com/keys)
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [SERP API Key](https://serpapi.com/manage-api-key)

## Deployment Steps

### 1. Connect Repository to Render

1. Log in to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select the MISAI repository
4. Render will auto-detect the `render.yaml` configuration

### 2. Configure Environment Variables

In the Render dashboard, add these environment variables:

| Variable Name | Description | Required |
|--------------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `GROQ_API_KEY` | Groq API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `SERP_API_KEY` | SERP API key for fact-checking | Yes |
| `VITE_BACKEND_HOST_URL` | Leave **empty** (same origin) | No |

> **Note**: `VITE_BACKEND_HOST_URL` should be left empty since the frontend and backend are served from the same domain.

### 3. Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Install Python dependencies
   - Install Node.js dependencies
   - Build the React frontend
   - Start the Gunicorn server
3. Wait for deployment to complete (5-10 minutes)

### 4. Access Your Application

Once deployed, Render will provide a URL like:
```
https://misai-app.onrender.com
```

Visit this URL to access your application!

## Build Process

The `render.yaml` file defines the build process:

```yaml
buildCommand: |
  pip install -r Backend/requirements.txt &&
  cd Frontend &&
  npm install &&
  npm run build &&
  cd ..
```

This:
1. Installs Python dependencies
2. Installs Node.js dependencies
3. Builds the React app to `Frontend/dist`
4. Backend serves these static files

## Verification

After deployment, verify:

1. **Homepage Loads**: Visit your Render URL
2. **IntroScreen**: Should see the MISAI intro animation
3. **MisBot**: Navigate to MisBot and send a test message
4. **TestAI**: Check that the iframe loads
5. **API Docs**: Visit `https://your-app.onrender.com/docs` to see FastAPI documentation

## Troubleshooting

### Build Fails

**Issue**: Build command fails during npm install or build

**Solution**:
- Check build logs in Render dashboard
- Ensure `package.json` is in `Frontend/` directory
- Verify Node.js version compatibility

### Frontend Not Loading

**Issue**: API works but frontend shows 404

**Solution**:
- Verify `Frontend/dist` folder was created during build
- Check that `main.py` has static file mounting code
- Review server logs for errors

### API Endpoints Return 404

**Issue**: Backend API endpoints not accessible

**Solution**:
- Ensure routes are defined before the catch-all SPA route
- Check that API endpoints start with `/` (e.g., `/chatapi`, `/testai`)
- Verify CORS middleware is configured

### Environment Variables Not Working

**Issue**: API keys not being recognized

**Solution**:
- Double-check variable names in Render dashboard
- Ensure no extra spaces in values
- Restart the service after adding variables

## Free Tier Limitations

Render's free tier includes:
- ✅ 750 hours/month of runtime
- ⚠️ Services spin down after 15 minutes of inactivity
- ⚠️ Cold starts take 30-60 seconds

**Tip**: Upgrade to paid tier for always-on service and faster performance.

## Updating Your Application

To deploy updates:

1. Push changes to your GitHub repository
2. Render will automatically detect changes
3. Auto-deploy will trigger (if enabled)
4. Or manually click **"Deploy latest commit"** in dashboard

## Support

- **Render Docs**: [docs.render.com](https://docs.render.com)
- **FastAPI Docs**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Vite Docs**: [vitejs.dev](https://vitejs.dev)

---

**Congratulations!** 🎉 Your MISAI application is now live on Render!
