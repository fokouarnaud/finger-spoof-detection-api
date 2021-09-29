# app.py#Import necessary packages
from base import Candidate,Classroomsubjectclass,classroomsubjectclasscandidate, db
from flask import Flask
from flask_restful import Resource, reqparse, Api  # Instantiate a flask object
#import numpy as np
#import pickle

app = Flask(__name__)

# Instantiate Api object
api = Api(app)

# Setting the location for the sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_one.db'

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
#model = pickle.load(open("model.pkl", "rb"))

class Candidate_List(Resource):

    # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('keypoints', type=str, required=False,
                        help='keypoints of the candidate')
    parser.add_argument('descriptors', type=str, required=False, help='descriptions of the candidate')
    

    # Creating the get method
    def get(self, name):
        item = Candidate.find_by_name(name)
        if item:
            return item.json()
        return {'Message': 'Candidate is not found'}

    # Creating the post method
    def post(self, name):
        if Candidate.find_by_name(name):
            return {' Message': 'Candidate with the  name {} already exists'.format(name)}

        args = Candidate_List.parser.parse_args()
        item = Candidate(name,args['keypoints'],
                      args['descriptors'])

        item.save_to()
        return item.json()

    # Creating the put method
    def put(self, name):
        args = Candidate_List.parser.parse_args()
        item = Candidate.find_by_name(name)
        if item:
            item.keypoints = args['keypoints']
            item.descriptors = args['descriptors']
            item.save_to()
            return {'Candidate': item.json()}
        item = Candidate(name, args['keypoints'],
                      args['descriptors'])
        item.save_to()
        return item.json()  # Creating the delete method

    def delete(self, name):
        item = Candidate.find_by_name(name)
        if item:
            item.delete_()
            return {'Message': '{} has been deleted from records'.format(name)}
        return {'Message': '{} is already not on the list'.format(name)}


# Creating a class to get all the movies from the database.
class All_Candidate(Resource):  # Defining the get method
    def get(self):
        # Adding the URIs to the api
        return {'Candidates': list(map(lambda x: x.json(), Candidate.query.all()))}


class Classroomsubjectclass_List(Resource):

    # Creating the get method
    def get(self, name):
        item = Classroomsubjectclass.find_by_name(name)
        if item:
            return item.json()
        return {'Message': 'Classroom subject class is not found'}

    # Creating the post method
    def post(self, name):
        if Classroomsubjectclass.find_by_name(name):
            return {' Message': 'Classroom subject class with the  name {} already exists'.format(name)}

       
        item = Classroomsubjectclass(name)

        item.save_to()
        return item.json()


    def delete(self, name):
        item = Classroomsubjectclass.find_by_name(name)
        if item:
            item.delete_()
            return {'Message': '{} has been deleted from records'.format(name)}
        return {'Message': '{} is already not on the list'.format(name)}


# Creating a class to get all the movies from the database.
class All_Classroomsubjectclass(Resource):  # Defining the get method
    def get(self):
        # Adding the URIs to the api
        return {'Classroomsubjectclass': list(map(lambda x: x.json(), Classroomsubjectclass.query.all()))}


# Creating a class to get iris prediction


#class IrisPredict(Resource):  # Defining the get method
    # Instantiating a parser object to hold data from message payload
#    parser = reqparse.RequestParser()
#    parser.add_argument('sepal_length', type=str, required=True,
#                        help='sepal_length')
#    parser.add_argument('sepal_width', type=str, required=True,
#                        help='sepal_width')
#    parser.add_argument('petal_length', type=str, required=True,
#                        help='petal_length')
#    parser.add_argument('petal_width', type=str, required=True,
#                        help='petal_width')

#    def get(self):
#        args = IrisPredict.parser.parse_args()
#        float_features = [
#            float(args.get("sepal_length")),
#            float(args.get("sepal_width")),
#            float(args.get("petal_length")),
#            float(args.get("petal_width"))
#        ]

#        print(float_features)
#        features = [np.array(float_features)]
#        prediction = model.predict(features)
#        return {'prediction_text':  "The flower species is {}".format(prediction)}


api.add_resource(All_Candidate, '/candidate',)
api.add_resource(Candidate_List, '/candidate/<string:name>')
api.add_resource(All_Classroomsubjectclass, '/classroomsubjectclass')
api.add_resource(Classroomsubjectclass_List, '/classroomsubjectclass/<string:name>')

#api.add_resource(IrisPredict, '/iris')

if __name__ == '__main__':
    # Run the applications
    app.run()
