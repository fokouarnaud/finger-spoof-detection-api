# app.py#Import necessary packages
from base import Candidate,Classroomsubjectclass,Classroomsubjectclasscandidat, db
from flask import Flask
from flask_restful import Resource, reqparse, Api  # Instantiate a flask object
from imageio import imread
import base64
import io
import cv2
import json

from fingerphoto.utils import *
from fingerphoto.utils2 import *
from fingerphoto.constants import *
import fingerphoto.fingerprint_feature_extractor

#import numpy as np
#import pickle

app = Flask(__name__)

# Instantiate Api object
api = Api(app)

# Setting the location for the sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_one_v3.db'

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


class CandidateClassroomsubjectclassListAPI(Resource):

    # Creating the get method
    def get(self, id):
        item = Classroomsubjectclass.find_by_id(id)
        if item:
            return {'Candidates': list(map(lambda x: x.json(), item.classroomsubjectclasscandidates))}
        return {'Message': 'Classroom subject class is not found'}
    
    
class CandidateClassroomsubjectclassAPI(Resource):
    # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('is_present', type=int, required=False,
                        help='presence of the candidate in classroom')

    # Creating the get method
    def get(self, classroom_id,candidat_id):
        item = Classroomsubjectclass.find_by_id(classroom_id)
        if item:
            itemCandidat = Candidate.find_by_id(candidat_id)
            if itemCandidat:
                assoc = Classroomsubjectclasscandidat.query.filter_by(candidate_id=candidat_id, classroom_subject_class_id=classroom_id).first()
                return {'Classroomsubjectclasscandidat':  assoc.json()}
            return {'Message': 'Candidate is not found'}
           
        return {'Message': 'Classroom subject class is not found'}
    
   # Creating the get method
    def post(self,classroom_id,candidat_id):

       
        item = Classroomsubjectclass.find_by_id(classroom_id)
        if item:
            itemCandidat = Candidate.find_by_id(candidat_id)
            if itemCandidat:
                assoc = Classroomsubjectclasscandidat(classroom_id,candidat_id)
                assoc.candidate= itemCandidat
                item.classroomsubjectclasscandidates.append(assoc)
                db.session.commit()
                return {'Candidates': list(map(lambda x: x.json(), item.classroomsubjectclasscandidates))}
            return {'Message': 'Candidate is not found'}
           
        return {'Message': 'Classroom subject class is not found'}

    # Creating the put method
    def put(self,classroom_id,candidat_id):
        args = CandidateClassroomsubjectclassAPI.parser.parse_args()
        item = Classroomsubjectclass.find_by_id(classroom_id)
        if item:
            itemCandidat = Candidate.find_by_id(candidat_id)
            if itemCandidat:
                assoc = Classroomsubjectclasscandidat.query.filter_by(candidate_id=candidat_id, classroom_subject_class_id=classroom_id).first()
                assoc.is_present=args['is_present']
                assoc.save_to()
                return {'Classroomsubjectclasscandidat':  assoc.json()}
            return {'Message': 'Candidate is not found'}
        return {'Message': 'Classroom subject class is not found'}


    def delete(self, id):
        item = Classroomsubjectclass.find_by_id(id)
        if item:
            item.delete_()
            return {'Message': '{} has been deleted from records'.format(id)}
        return {'Message': '{} is already not on the list'.format(id)}



api.add_resource(CandidateClassroomsubjectclassListAPI, '/classroomsubjectclasses/<int:id>/candidates',endpoint='Candidates_by_classroomsubjectclass')
api.add_resource(CandidateClassroomsubjectclassAPI, '/classroomsubjectclasses/<int:classroom_id>/candidates/<int:candidat_id>',endpoint='candidate_to_classroomsubjectclass')


    
class FingerphotoProcessingAPI(Resource):
    # Instantiating a parser object to hold data from message payload
    parser = reqparse.RequestParser()
    parser.add_argument('action', type=str, required=False,
                        help='action in authentication process')
    parser.add_argument('img', type=str, required=False,
                        help='img of fingerphoto')
   # Creating the get method
    def post(self):

        args = FingerphotoProcessingAPI.parser.parse_args()
        b64_string=args['img']
        # reconstruct image as an numpy array
        img = imread(io.BytesIO(base64.b64decode(b64_string)))

       

        # finally convert RGB image to BGR for opencv
        # and save result
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        img= skinDetection(cv2_img)
        #gray_img = grayscale_image(img)
        padding,img = enhance_image_target(img)
        #img = enhance_image(img)
        img = thinning(img)

        # show image
        #plt.figure()
        #plt.imshow(img, cmap="gray")
        #plt.show()
        b64_string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
       
       # Initiate ORB detector for matching keypoints
        orb = cv2.ORB_create(MAX_FEATURES)
        kp, des = get_feature_keypoint_and_descriptor(img, orb,padding)
        kp_json =json.dumps([{'x':k.pt[0],'y':k.pt[1], 'size':k.size,'octave':k.octave,'class_id':k.class_id,'angle': k.angle, 'response': k.response} for k in kp])
        des_json=json.dumps(des.tolist())

        return {'action': args['action'],
            'img': b64_string,
            'keypoints':kp_json,
            'descriptions':des_json
            }
       

api.add_resource(FingerphotoProcessingAPI, '/fingerphoto',endpoint='fingerphoto_processing')

if __name__ == '__main__':
    # Run the applications
    app.run()
