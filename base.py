
# base.py
# #Importing sqlalchemy
from flask_sqlalchemy import SQLAlchemy
# Instantiating sqlalchemy object
db = SQLAlchemy()  # Creating database class


#classroomsubjectclasscandidate = db.Table('classroomsubjectclasscandidate',
#    db.Column('candidate_id',db.Integer,db.ForeignKey('candidate.candidate_id')),
#    db.Column('classroom_subject_class_id',db.Integer,db.ForeignKey('classroomsubjectclass.classroom_subject_class_id')),
#    db.Column('is_present',db.Boolean, unique=False, default=False)
#)

class Classroomsubjectclasscandidat(db.Model):
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.candidate_id'),
                              primary_key=True)
    classroom_subject_class_id = db.Column(db.Integer, db.ForeignKey('classroomsubjectclass.classroom_subject_class_id'),
                               primary_key=True)
    is_present=db.Column(db.Integer)
    
    candidate= db.relationship('Candidate')

    def __init__(self, candidate_id, classroom_subject_class_id):
        self.candidate_id = candidate_id
        self.classroom_subject_class_id = classroom_subject_class_id
        self.is_present=0

    # Method to show data as dictionary object
    def json(self):
        return {
            'candidate_id':self.candidate_id,
            'classroom_subject_class_id':self.classroom_subject_class_id,
            'is_present': self.is_present}


    # Method to find the query element is existing or not
    @classmethod
    def find_by_id(cls, candidate_id,classroom_subject_class_id):
        return cls.query.filter_by(candidate_id=candidate_id,classroom_subject_class_id=classroom_subject_class_id).first()

 
    # Method to save data to database
    def save_to(self):
        db.session.add(self)
        db.session.commit()

    # Method to delete data from database
    def delete_(self):
        db.session.delete(self)
        db.session.commit()


class Candidate(db.Model):

    # Creating field/columns of the database as class variables
    candidate_id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(30), unique=True, nullable=False)

    keypoints = db.Column(db.String(10005760), unique=False, nullable=False)

    descriptors = db.Column(db.String(10005760), unique=False, nullable=False)

 

    def __init__(self, name, keypoints, descriptors):
        self.name = name
        self.keypoints = keypoints
        self.descriptors = descriptors

    # Method to show data as dictionary object
    def json(self):
        return {
            'candidate_id':self.candidate_id,
            'name': self.name,
                'keypoints': self.keypoints,
                'descriptors': self.descriptors}

    # Method to find the query element is existing or not
    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(candidate_id=id).first()

    # Method to find the query element is existing or not
    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name=name).first()

    # Method to save data to database
    def save_to(self):
        db.session.add(self)
        db.session.commit()

    # Method to delete data from database
    def delete_(self):
        db.session.delete(self)
        db.session.commit()


class Classroomsubjectclass (db.Model):

    # Creating field/columns of the database as class variables
    classroom_subject_class_id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(30), unique=True, nullable=False)
    classroomsubjectclasscandidates = db.relationship('Classroomsubjectclasscandidat')

 

    def __init__(self, name):
        self.name = name

    # Method to show data as dictionary object
    def json(self):
        return {
            'classroom_subject_class_id':self.classroom_subject_class_id,
            'name': self.name}

    # Method to find the query element is existing or not
    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(classroom_subject_class_id=id).first()

    # Method to find the query element is existing or not
    @classmethod
    def find_by_name(cls, name):
        return cls.query.filter_by(name=name).first()

    # Method to save data to database
    def save_to(self):
        db.session.add(self)
        db.session.commit()

    # Method to delete data from database
    def delete_(self):
        db.session.delete(self)
        db.session.commit()
