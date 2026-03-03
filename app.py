import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import tempfile
from bottle import Bottle, request, template, run

from predict_disorder_simpler import main as predict_main

app = Bottle()


@app.route("/")
def index():
    return template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict Disorder</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .form-container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            margin-top: 0;
            color: #333;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #444;
        }
        input[type="text"],
        select,
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .checkbox-group input {
            width: auto;
        }
        .checkbox-group label {
            margin: 0;
            font-weight: normal;
        }
        .help-text {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .results {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .result-item {
            margin: 10px 0;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
        .success {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h1>Predict Disorder</h1>
        <form id="predictForm" action="/predict" method="POST" enctype="multipart/form-data">
            <div class="form-group">
                <label for="protein_id">Select Protein ID</label>
                <select id="protein_id" name="protein_id" required>
                    <option value="" disabled selected>Select a protein</option>
                    <option value="DP04142">DP04142</option>
                    <option value="DP04179">DP04179</option>
                    <option value="DP04199">DP04199</option>
                    <option value="DP04219">DP04219</option>
                </select>
            </div>

            <button type="submit">Run Prediction</button>
        </form>
    </div>
</body>
</html>""")


@app.route("/predict", method="POST")
def predict():
    protein_id = request.forms.get("protein_id")

    if not protein_id:
        return template("""
            <div class="error">No protein ID selected</div>
            <a href="/">Go back</a>
        """)

    try:
        import traceback

        stats = predict_main(protein_id=protein_id)

        plot_filename = f"{protein_id}_ESM2_plot.png"
        plot_url = f"/static/{plot_filename}"

        # Format the percentage for display
        disorder_percentage_formatted = f"{stats['disorder_percentage']:.2f}"

        return template(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .results {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            margin-top: 0;
            color: #333;
        }
        .result-item {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .result-item strong {
            color: #007bff;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .plot-container {
            margin: 30px 0;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }
        .plot-container img {
            display: block;
            max-width: 100%;
            width: auto;
            height: auto;
            margin: 0 auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .button-container {
            text-align: center;
            margin-top: 30px;
        }
        a.button {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border-radius: 4px;
            text-decoration: none;
            transition: background-color 0.2s;
        }
        a.button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="results">
        <h1>Disorder Prediction Results</h1>
        
        <div class="result-item">
            <strong>Protein ID:</strong> {{protein_id}}
        </div>
        
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Residues</div>
                <div class="stat-value">{{total_residues}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Disordered Residues</div>
                <div class="stat-value">{{disordered_residues}}</div>
            </div>
        </div>
        
        <div class="stat-card" style="background: #e8f5e9; margin: 20px 0;">
            <div class="stat-label" style="color: #2e7d32;">Disorder Percentage</div>
            <div class="stat-value" style="color: #2e7d32; font-size: 36px;">{{disorder_percentage_formatted}}%</div>
        </div>
        
        <div class="plot-container">
            <h2 style="color: #333; font-size: 18px;">Disorder Prediction Plot</h2>
            <img src="{{plot_url}}" alt="Disorder Prediction Plot for {{protein_id}}">
        </div>
        
        <div class="button-container">
            <a href="/" class="button">Run Another Prediction</a>
        </div>
    </div>
</body>
</html>""",
            protein_id=stats["protein_id"],
            total_residues=stats["total_residues"],
            disordered_residues=stats["disordered_residues"],
            disorder_percentage_formatted=disorder_percentage_formatted,
            plot_url=plot_url,
        )

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return template(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Prediction Failed</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .error-container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 4px;
            border-left: 4px solid #c62828;
            margin: 20px 0;
        }
        h1 {
            color: #c62828;
            margin-top: 0;
        }
        pre {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.4;
        }
        .button-container {
            text-align: center;
            margin-top: 30px;
        }
        a.button {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border-radius: 4px;
            text-decoration: none;
            transition: background-color 0.2s;
        }
        a.button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>Prediction Failed</h1>
        <div class="error">
            <strong>Error:</strong> {{error}}
        </div>
        <details>
            <summary style="cursor: pointer; font-weight: bold; margin: 20px 0;">Show detailed traceback</summary>
            <pre>{{traceback}}</pre>
        </details>
        <div class="button-container">
            <a href="/" class="button">Go Back</a>
        </div>
    </div>
</body>
</html>""",
            error=str(e),
            traceback=traceback_str,
        )


@app.route("/static/<filename>")
def serve_static(filename):
    from bottle import static_file

    return static_file(filename, root="results_prueba")


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8080, debug=True)
