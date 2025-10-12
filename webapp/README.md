# Bitcoin Price Prediction - Web Application

A Flask-based web application showcasing the Bitcoin price prediction project with 3 main pages:

1. **Methodology**: Feature engineering and return-based prediction approach
2. **Test Results**: Model performance metrics and validation
3. **Live Performance**: Real-world predictions with blockchain integration

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you're in the project root
cd /Users/ying-jeanne/Workspace/capstone_bitcoin

# Activate virtual environment (if not already active)
source venv/bin/activate

# Install/update requirements
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Optional)

If you haven't run the pipeline yet and want real results data:

```bash
python run_full_pipeline.py
```

This will:
- Fetch latest Bitcoin data
- Engineer features
- Train models
- Generate results

### 3. Start the Web Application

```bash
python webapp/app.py
```

The server will start on `http://localhost:5000`

### 4. Access the Website

Open your browser and visit:
- **Home/Methodology**: http://localhost:5000/
- **Test Results**: http://localhost:5000/results
- **Live Performance**: http://localhost:5000/live

## Project Structure

```
webapp/
├── app.py                  # Flask application (main entry point)
├── templates/              # HTML templates
│   ├── base.html          # Base template with navigation
│   ├── methodology.html   # Page 1: Features & methodology
│   ├── results.html       # Page 2: Test results
│   └── live.html          # Page 3: Live performance
└── static/                # Static assets
    ├── css/
    │   └── style.css      # Main stylesheet
    └── js/
        └── main.js        # JavaScript utilities
```

## Features

### Page 1: Methodology
- **Innovation Showcase**: Visual explanation of return-based prediction
- **Feature Explorer**: Interactive display of all 55+ features with formulas
- **Model Configuration**: Hyperparameters and training setup
- **Pipeline Visualization**: Step-by-step data flow

### Page 2: Test Results
- **Metrics Dashboard**: MAPE, R², MAE, Directional Accuracy
- **Interactive Charts**: Comparison charts using Chart.js
- **Model Rankings**: Best to worst performing models
- **Bias Diagnosis**: Verification of bias elimination
- **Key Insights**: Actionable takeaways from results

### Page 3: Live Performance
- **Current Price Display**: Real-time Bitcoin price with auto-refresh
- **Test vs Live Comparison**: Consistency verification
- **Blockchain Integration**: Display of on-chain predictions
- **Historical Charts**: Predictions vs actual outcomes over time
- **Performance Metrics**: Rolling MAPE and directional accuracy

## API Endpoints

The app provides REST API endpoints for programmatic access:

```bash
# Get latest Bitcoin price
curl http://localhost:5000/api/latest-price

# Get all model results
curl http://localhost:5000/api/model-results

# Get blockchain predictions
curl http://localhost:5000/api/blockchain-predictions

# Get feature definitions
curl http://localhost:5000/api/feature-definitions

# Get project statistics
curl http://localhost:5000/api/stats
```

## Mock Data

The application uses **mock data** for blockchain predictions until the smart contract is deployed. This allows you to see the full functionality immediately.

To integrate with a real smart contract:
1. Follow [SMART_CONTRACT_PLAN.md](../SMART_CONTRACT_PLAN.md)
2. Deploy contract to Polygon
3. Update `webapp/app.py` to use `utils/blockchain_integration.py`
4. Replace `get_blockchain_predictions()` with actual contract calls

## Customization

### Changing Port

Edit `webapp/app.py` line 305:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### Updating Styles

Edit `webapp/static/css/style.css` to customize colors, fonts, layouts.

CSS variables are defined at the top:
```css
:root {
    --primary-color: #1f77b4;
    --secondary-color: #ff7f0e;
    /* ... */
}
```

### Adding New Pages

1. Create template in `webapp/templates/`
2. Add route in `webapp/app.py`:
```python
@app.route('/my-new-page')
def my_new_page():
    return render_template('my-new-page.html')
```
3. Add navigation link in `base.html`

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
python webapp/app.py --port 5001
```

### Import Errors

Make sure you're running from the project root and virtual environment is activated:

```bash
cd /Users/ying-jeanne/Workspace/capstone_bitcoin
source venv/bin/activate
python webapp/app.py
```

### No Results Data

If you see "Results not available" on the Test Results page:

```bash
# Run the full pipeline to generate results
python run_full_pipeline.py
```

### Charts Not Displaying

Make sure Chart.js is loading. Check browser console (F12) for errors. The CDN link is in `base.html`:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

## Development Mode

The app runs in debug mode by default, which provides:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger in browser

For production deployment, set `debug=False` in `app.py`.

## Next Steps

1. **Run the app**: `python webapp/app.py`
2. **Explore all 3 pages**: Navigate through methodology → results → live
3. **Check API endpoints**: Test the REST API with curl or browser
4. **Review smart contract plan**: Read [SMART_CONTRACT_PLAN.md](../SMART_CONTRACT_PLAN.md)
5. **Deploy blockchain integration**: Follow the 5-week timeline in the plan

## Presentation Tips

When presenting to your group:

1. **Start with Page 1**: Explain the key innovation (return-based prediction)
2. **Show feature engineering**: Highlight the 55+ features with formulas
3. **Demo test results**: Show the impressive metrics (1.16% MAPE, 0.865 R²)
4. **Emphasize bias fix**: Compare before/after results
5. **Show blockchain vision**: Explain Page 3 and the smart contract plan
6. **Demo live features**: Show auto-refreshing price and charts

## Support

For questions about:
- **Flask app**: Check this README or Flask documentation
- **Smart contracts**: See [SMART_CONTRACT_PLAN.md](../SMART_CONTRACT_PLAN.md)
- **Model training**: See [CLAUDE.md](../CLAUDE.md)
- **General project**: See [run_full_pipeline.py](../run_full_pipeline.py)

---

**Built with**: Flask, Chart.js, Python 3.13
**Deployment ready**: Yes (set debug=False for production)
**Mobile responsive**: Yes
**Browser support**: All modern browsers
