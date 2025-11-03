# Shark Tank AI Analyzer
<div align="center">

<a href="https://peerlist.io/spandanm110/project/shark-tank-gpt" target="_blank" rel="noreferrer">
    <img
        src="https://peerlist.io/api/v1/projects/embed/PRJHR8DDDMGG7L7OLIANLBRJA6BBOL?showUpvote=true&theme=dark"
        alt="Shark Tank GPT"
        style="width: auto; height: 72px;"
    />
</a>

</div>
A comprehensive AI-powered analysis system for Shark Tank pitches using LangGraph and Streamlit. This system provides intelligent investment analysis, pitch evaluation, and success prediction based on historical Shark Tank data from US, India, and Australia.

## Features

### Multi-Agent Analysis System
- Query Classifier: Routes queries to appropriate analysis paths
- Country Analyzer: Analyzes patterns across different countries
- Shark Profiler: Profiles individual shark investment strategies
- Industry Analyzer: Identifies hot industries and success patterns
- Pitch Evaluator: Evaluates specific pitches and ideas
- ML Analyzer: Advanced machine learning predictions
- Success Predictor: Predicts deal success probability
- Recommendation Engine: Generates actionable recommendations

### Advanced Analytics
- Machine Learning Models: Random Forest classifier for success prediction
- Interactive Visualizations: Plotly charts for data exploration
- Real-time Analysis: Live processing of pitch queries
- Comprehensive Reports: Detailed markdown reports with insights

### Interface
- Natural Language Queries: Ask questions in plain English
- File Upload Support: Upload pitch decks, business plans, or data files
- Chat History: Track all previous analyses
- Export Functionality: Download reports and analysis data

## Project Structure

```
Sharktank_GPT_Streamlit/
├── streamlit_app.py              # Main Streamlit application
├── langgraph_workflow.py         # LangGraph workflow implementation
├── groq_integration.py           # Groq LLM integration
├── advanced_analysis.py          # ML and advanced analytics
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── Shark Tank US dataset.csv     # US dataset
├── Shark Tank India.csv          # India dataset
├── Shark Tank Australia dataset.csv  # Australia dataset
└── shark_tank_merged.csv         # Merged dataset
```

## Prerequisites

- Python 3.8 or higher
- Required CSV dataset files in the project directory
- Groq API key (get from https://console.groq.com/)

## Installation

### Option 1: Direct Installation

1. Clone or download the project files:
   ```bash
   git clone https://github.com/yourusername/Sharktank_GPT_Streamlit.git
   cd Sharktank_GPT_Streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here  # Optional
   LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here  # Optional
   LANGFUSE_BASE_URL=https://cloud.langfuse.com    # Optional
   LANGFUSE_ENABLED=false                          # Optional
   ```

4. Ensure all CSV dataset files are in the project root directory:
   - `Shark Tank US dataset.csv`
   - `Shark Tank India.csv`
   - `Shark Tank Australia dataset.csv`
   - `shark_tank_merged.csv`

5. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

6. Open your browser to `http://localhost:8501`

### Option 2: Virtual Environment (Recommended)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Follow steps 3-6 from Option 1

## Configuration

### Environment Variables

The app requires environment variables to be set. Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGFUSE_SECRET_KEY=sk-lf-...  # Optional, for observability
LANGFUSE_PUBLIC_KEY=pk-lf-...  # Optional, for observability
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Optional
LANGFUSE_ENABLED=false  # Set to true to enable Langfuse
```

**Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

**Langfuse Setup (Optional)**: To enable observability with Langfuse:
1. Sign up at https://cloud.langfuse.com
2. Create a new project
3. Get your API keys from Settings → API Keys
4. Add them to your `.env` file and set `LANGFUSE_ENABLED=true`

### Customization

Edit `config.py` to customize:
- Analysis thresholds
- Visualization colors
- File upload limits
- Report settings

## Usage Examples

### Sample Queries

1. Pitch Analysis:
   ```
   "I want to pitch a food tech startup asking for $500k for 15% equity"
   ```

2. Industry Research:
   ```
   "What are the most successful industries in Shark Tank?"
   ```

3. Shark Comparison:
   ```
   "Compare investment patterns between US and India sharks"
   ```

4. Success Prediction:
   ```
   "Analyze my pitch: AI-powered fitness app, $1M ask, 20% equity"
   ```

### File Upload
- Upload CSV files with pitch data
- Upload text files with business descriptions
- Upload markdown files with pitch decks

## Analysis Features

### Success Prediction
- Probability Score: 0-100% success likelihood
- Confidence Level: Model confidence in prediction
- Success Level: High/Medium/Low classification
- ML Models: Random Forest with feature importance

### Investment Patterns
- Country Analysis: Success rates by country
- Industry Trends: Hot industries and success patterns
- Shark Profiles: Individual investment strategies
- Gender Analysis: Investment patterns by gender

### Risk Assessment
- Equity Analysis: Optimal equity ranges
- Valuation Checks: Reasonable ask amounts
- Industry Risks: Sector-specific challenges
- Market Factors: External risk considerations

### Visualizations
- Interactive Charts: Plotly-powered visualizations
- Country Comparison: Success rates and metrics
- Industry Analysis: Performance by sector
- Shark Profiles: Investment patterns
- Trend Analysis: Historical patterns

### Reports
- Executive summary with key metrics
- Detailed country and industry analysis
- Shark investment profiles
- Success factors and risk assessment
- Actionable recommendations
- Downloadable markdown format

## Deployment to Streamlit Cloud

### Prerequisites
- GitHub account with repository set up
- Streamlit Cloud account (sign up at https://share.streamlit.io)
- Groq API key

### Step 1: Push Code to GitHub

1. Initialize git repository (if not already done):
   ```bash
   git init
   ```

2. Add all files:
   ```bash
   git add .
   ```

3. Commit changes:
   ```bash
   git commit -m "Ready for Streamlit deployment"
   ```

4. Create a new repository on GitHub (if not exists):
   - Go to https://github.com and sign in
   - Click "+" icon → "New repository"
   - Repository name: `Sharktank_GPT_Streamlit`
   - Choose Public (for free Streamlit Cloud) or Private
   - DO NOT initialize with README, .gitignore, or license
   - Click "Create repository"

5. Connect and push to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Sharktank_GPT_Streamlit.git
   git branch -M main
   git push -u origin main
   ```

**Note**: If you get authentication errors, use a GitHub Personal Access Token:
- GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate new token with `repo` permissions
- Use token as password when pushing

### Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub

2. Click "New app"

3. Configure the app:
   - Select your repository: `Sharktank_GPT_Streamlit`
   - Select branch: `main` (or `master`)
   - Main file path: `streamlit_app.py`

4. Add Secrets (API keys):
   - Go to app settings → Secrets
   - Add your environment variables:
     ```toml
     GROQ_API_KEY = "your_groq_api_key_here"
     LANGFUSE_SECRET_KEY = "sk-lf-..."  # Optional
     LANGFUSE_PUBLIC_KEY = "pk-lf-..."  # Optional
     LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
     LANGFUSE_ENABLED = "false"
     ```

5. Click "Deploy" and wait 2-5 minutes

6. Your app will be live at: `https://your-app-name.streamlit.app`

### Important Notes for Deployment

#### Dataset Files
Make sure your CSV files are committed to the repository:
- `Shark Tank US dataset.csv`
- `Shark Tank India.csv`
- `Shark Tank Australia dataset.csv`
- `shark_tank_merged.csv`

These files should be in the root directory of your repository and are required for the app to function.

#### Requirements.txt
Your `requirements.txt` file is already configured with all necessary dependencies. Streamlit Cloud will automatically install them.

#### Memory and Performance
- Streamlit Cloud provides free tier with 1GB RAM
- For heavy ML workloads, consider upgrading to paid tier
- The app loads datasets at startup, so initial load may take a few seconds

#### Environment Variables
The app uses `.env` files locally, but on Streamlit Cloud, use the Secrets feature instead. The `python-dotenv` package will read from Streamlit secrets automatically.

### Updating Your App

To update your deployed app:
1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update description"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy
4. You can also manually trigger redeploy from the app settings

## Troubleshooting

### Common Issues

1. Import Errors:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. Groq API Errors:
   - Check your internet connection
   - Verify the API key in `.env` file or Streamlit Cloud secrets
   - Get your API key from https://console.groq.com/

3. File Not Found Errors:
   - Verify CSV files are in the repository root directory
   - Check file names match exactly (case-sensitive)
   - Ensure files are committed to GitHub

4. API Key Errors:
   - Verify secrets are set correctly in Streamlit Cloud
   - Check that keys don't have extra spaces or quotes
   - Ensure `.env` file exists locally with correct keys

5. Memory Issues:
   - Dataset files might be too large for free tier
   - Consider using data caching (already implemented in the code)
   - Close other applications if running locally

6. Port Already in Use:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

7. Virtual Environment Issues:
   - Ensure virtual environment is activated
   - Reinstall packages: `pip install -r requirements.txt`
   - Verify Python version: `python --version` (should be 3.8+)

8. Git Authentication Issues:
   - Use GitHub Personal Access Token instead of password
   - Or use SSH: `git@github.com:YOUR_USERNAME/Sharktank_GPT_Streamlit.git`

### Checking Logs (Streamlit Cloud)

1. Go to your app in Streamlit Cloud
2. Click the menu (three dots) in the top right
3. Select "Manage app"
4. View logs for error messages

### Large Files

If CSV files are too large (>100MB):
- Consider using Git LFS: `git lfs install && git lfs track "*.csv"`
- Or upload datasets to cloud storage and load from URL

## Security Best Practices

**DO**:
- Use Streamlit Secrets for all API keys on Streamlit Cloud
- Keep your `.gitignore` updated
- Never commit `.env` files
- Review your code before pushing to GitHub
- Use environment variables instead of hardcoded values

**DON'T**:
- Hardcode API keys in your code
- Commit sensitive data to GitHub
- Share your repository secrets publicly
- Use production API keys in development

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (recommended)
- Internet connection for Groq API
- Modern web browser

## Dependencies

All required packages are listed in `requirements.txt`:
- streamlit
- pandas
- numpy
- plotly
- langgraph
- langchain
- langchain-groq
- groq
- scikit-learn
- xgboost
- shap
- python-dotenv
- langfuse

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments
3. Open an issue on GitHub
4. Check Streamlit Cloud documentation: https://docs.streamlit.io/streamlit-cloud
5. Visit Streamlit Community: https://discuss.streamlit.io

## Future Enhancements

- Real-time data updates
- Additional ML models
- API integration
- Mobile app version
- Advanced NLP features
- Custom dashboard creation

---

Built with LangGraph, Streamlit, and Python
