from flask import Flask, render_template, request
import functions  # Import your notebook functions
import pandas as pd
import os

app = Flask(__name__)

# Set a secret key for session management (important for security)
app.secret_key = os.urandom(24)  # Generate a random secret key


@app.route("/", methods=["GET", "POST"])
def index():
    map_html = None  # Initialize map_html
    selected_date = None

    if request.method == "POST":
        selected_date_str = request.form["date"]
        try:
            # Convert the selected date string to a datetime object
            selected_date = pd.to_datetime(selected_date_str)
        except ValueError:
            return render_template("index.html", error="Invalid date format.  Please use YYYY-MM-DD.")
        
        # Generate or update the map
        map_html = functions.generate_map_for_date(selected_date)

        if map_html:
             # Save the map to the static folder (for GitHub Pages)
            with open("static/index.html", "w", encoding="utf-8") as f:
                f.write(map_html)


    # Use current day if not set yet
    if not selected_date:
        selected_date = pd.to_datetime('today')


    return render_template("index.html", map_html=map_html, selected_date=selected_date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    app.run(debug=True)  # Use debug=True for development, False for production        