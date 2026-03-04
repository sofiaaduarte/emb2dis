import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import tempfile
from bottle import Bottle, request, template, run

from predict_disorder import main as predict_main

app = Bottle()


@app.route("/")
def index():
    return template("templates/predict_form")


@app.route("/predict", method="POST")
def predict():
    sequence = request.forms.get("sequence")
    language_model = request.forms.get("model", "ESM2")
    protein_id = "input"
    
    fasta_file = request.files.get("fasta_file")
    if fasta_file and fasta_file.filename:
        fasta_content = fasta_file.file.read().decode("utf-8")
        if fasta_content.startswith(">"):
            lines = fasta_content.strip().split("\n")
            protein_id = lines[0][1:].strip()
            sequence = ""
            for line in lines[1:]:
                if not line.startswith(">"):
                    sequence += line.strip()
                else:
                    break
        else:
            sequence = fasta_content.strip()

    if not sequence:
        return template("templates/error_missing")

    try:
        import traceback

        stats_list = predict_main(sequence=sequence, language_model=language_model, protein_id=protein_id)
        stats = stats_list[0] if stats_list else {}

        plot_filename = f"emb2dis_{protein_id}_{language_model}_plot.png"
        plot_url = f"/static/{plot_filename}"

        disorder_percentage_formatted = f"{stats.get('disorder_percentage', 0):.2f}"

        return template(
            "templates/results",
            language_model=language_model,
            total_residues=stats.get("total_residues", 0),
            disordered_residues=stats.get("disordered_residues", 0),
            disorder_percentage_formatted=disorder_percentage_formatted,
            plot_url=plot_url,
        )

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return template(
            "templates/error",
            error=str(e),
            traceback=traceback_str,
        )


@app.route("/static/<filename>")
def serve_static(filename):
    from bottle import static_file

    return static_file(filename, root="results")


@app.route("/download/<filename>")
def download_sample(filename):
    from bottle import static_file

    return static_file(filename, root="data", download=filename)


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8080, debug=True)
