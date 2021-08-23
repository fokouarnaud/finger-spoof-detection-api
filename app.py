# app.py#Import necessary packages
from base import Movies, db
from flask import Flask
from flask_restful import Resource, reqparse, Api  # Instantiate a flask object
import numpy as np
import pickle

app = Flask(__name__)

# Instantiate Api object
api = Api(app)

# Setting the location for the sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///base.db'

# Adding the configurations for the database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Import necessary classes from base.py
app.config['PROPAGATE_EXCEPTIONS'] = True

# Link the app object to the Movies database
db.init_app(app)
app.app_context().push()

# Create the databases
db.create_all()  # Creating a class to create get, post, put & delete methods


# load the pickle model
model = pickle.load(open("model.pkl", "rb"))


class Movies_List(Resource):

    # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('director', type=str, required=False,
                        help='Director of the movie')
    parser.add_argument('genre', type=str, required=False,
                        help='Genre of the movie')
    parser.add_argument('collection', type=int, required=True,
                        help='Gross collection of the movie')

    # Creating the get method
    def get(self, movie):
        item = Movies.find_by_title(movie)
        if item:
            return item.json()
        return {'Message': 'Movie is not found'}

    # Creating the post method
    def post(self, movie):
        if Movies.find_by_title(movie):
            return {' Message': 'Movie with the  title {} already exists'.format(movie)}

        args = Movies_List.parser.parse_args()
        item = Movies(movie, args['director'],
                      args['genre'],      args['collection'])

        item.save_to()
        return item.json()

    # Creating the put method
    def put(self, movie):
        args = Movies_List.parser.parse_args()
        item = Movies.find_by_title(movie)
        if item:
            item.collection = args['collection']
            item.save_to()
            return {'Movie': item.json()}
        item = Movies(movie, args['director'],
                      args['genre'], args['collection'])
        item.save_to()
        return item.json()  # Creating the delete method

    def delete(self, movie):
        item = Movies.find_by_title(movie)
        if item:
            item.delete_()
            return {'Message': '{} has been deleted from records'.format(movie)}
        return {'Message': '{} is already not on the list'.format()}


# Creating a class to get all the movies from the database.
class All_Movies(Resource):  # Defining the get method
    def get(self):
        # Adding the URIs to the api
        return {'Movies': list(map(lambda x: x.json(), Movies.query.all()))}

# Creating a class to get iris prediction


class IrisPredict(Resource):  # Defining the get method
    # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('sepal_length', type=str, required=True,
                        help='sepal_length')
    parser.add_argument('sepal_width', type=str, required=True,
                        help='sepal_width')
    parser.add_argument('petal_length', type=str, required=True,
                        help='petal_length')
    parser.add_argument('petal_width', type=str, required=True,
                        help='petal_width')

    def get(self):
        args = IrisPredict.parser.parse_args()
        float_features = [
            float(args.get("sepal_length")),
            float(args.get("sepal_width")),
            float(args.get("petal_length")),
            float(args.get("petal_width"))
        ]

        print(float_features)
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return {'prediction_text':  "The flower species is {}".format(prediction)}


api.add_resource(All_Movies, '/')
api.add_resource(Movies_List, '/<string:movie>')
api.add_resource(IrisPredict, '/iris')

if __name__ == '__main__':
    # Run the applications
    app.run()
