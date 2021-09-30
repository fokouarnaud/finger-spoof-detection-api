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

class CandidateAPI(Resource):

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

        args = CandidateAPI.parser.parse_args()
        item = Candidate(name,args['keypoints'],
                      args['descriptors'])

        item.save_to()
        return item.json()

    # Creating the put method
    def put(self, name):
        args = CandidateAPI.parser.parse_args()
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
class CandidateListAPI(Resource):  # Defining the get method
    def get(self):
        # Adding the URIs to the api
        return {'Candidates': list(map(lambda x: x.json(), Candidate.query.all()))}

api.add_resource(CandidateAPI, '/candidate/<string:name>',endpoint='candidate')
api.add_resource(CandidateListAPI, '/candidates',endpoint='candidates')


class ClassroomsubjectclassAPI(Resource):


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
class ClassroomsubjectclassListAPI(Resource):  # Defining the get method
    def get(self):
        # Adding the URIs to the api
        return {'Classroomsubjectclass': list(map(lambda x: x.json(), Classroomsubjectclass.query.all()))}


api.add_resource(ClassroomsubjectclassListAPI, '/classroomsubjectclasses',endpoint='classroomsubjectclasses')
api.add_resource(ClassroomsubjectclassAPI, '/classroomsubjectclass/<string:name>',endpoint='classroomsubjectclass')

#api.add_resource(IrisPredict, '/iris')


class CandidatesByClassroomsubjectclassAPI(Resource):

     # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('candidat_id', type=int, required=False,
                        help='candidat_id of the candidate')

    # Creating the get method
    def get(self, id):
        item = Classroomsubjectclass.find_by_id(id)
        if item:
            return {'Candidates': list(map(lambda x: x.json(), item.classroomsubjectclasscandidates))}
        return {'Message': 'Classroom subject class is not found'}
    
    # Creating the get method
    def post(self, id):

        args = CandidatesByClassroomsubjectclassAPI.parser.parse_args()
        item = Classroomsubjectclass.find_by_id(id)
        if item:
            itemCandidat = Candidate.find_by_id(args['candidat_id'])
            if itemCandidat:
                item.classroomsubjectclasscandidates.append(itemCandidat)
                item.save_to()
                return {'Candidates': list(map(lambda x: x.json(), item.classroomsubjectclasscandidates))}
            return {'Message': 'Candidate is not found'}
           
        return {'Message': 'Classroom subject class is not found'}


    def delete(self, id):
        item = Classroomsubjectclass.find_by_id(id)
        if item:
            item.delete_()
            return {'Message': '{} has been deleted from records'.format(id)}
        return {'Message': '{} is already not on the list'.format(id)}



api.add_resource(CandidatesByClassroomsubjectclassAPI, '/classroomsubjectclasses/<int:id>/candidates',endpoint='Candidates_by_classroomsubjectclass')



if __name__ == '__main__':
    # Run the applications
    app.run()
