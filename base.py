
# base.py
# #Importing sqlalchemy
from flask_sqlalchemy import SQLAlchemy
# Instantiating sqlalchemy object
db = SQLAlchemy()  # Creating database class


class Movies(db.Model):

    # Creating field/columns of the database as class variables
    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(db.String(30), unique=True, nullable=False)

    director = db.Column(db.String(30), unique=False, nullable=False)

    genre = db.Column(db.String(30), unique=False, nullable=False)

    collection = db.Column(db.Integer, unique=False, nullable=False)

    def __init__(self, title, director, genre, collection):
        self.title = title
        self.director = director
        self.genre = genre
        self.collection = collection

    # Method to show data as dictionary object
    def json(self):
        return {'Title': self.title,
                'Director': self.director,
                'Genre': self.genre,
                'Collection': self.collection}

    # Method to find the query movie is existing or not
    @classmethod
    def find_by_title(cls, title):
        return cls.query.filter_by(title=title).first()

    # Method to save data to database
    def save_to(self):
        db.session.add(self)
        db.session.commit()

    # Method to delete data from database
    def delete_(self):
        db.session.delete(self)
        db.session.commit()
