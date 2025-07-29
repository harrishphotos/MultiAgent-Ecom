# The main entry point that runs our Flask web server 

from flask import Flask, render_template, request
from router import route_query
 
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])

def index():
    response = None
    if request.method == "POST":
        user_query = request.form['query']
        response = route_query(user_query)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    # app.run() starts the Flask development web server.
    # debug=True is very helpful during development. It automatically reloads the server
    # when you save a file and shows detailed error pages if something goes wrong.
    app.run(debug=True, port=8002)



