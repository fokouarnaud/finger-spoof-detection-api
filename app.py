from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

# Creaating a empty database for storing the products
SHOPPING_CART = {}

# adding arguments needed for parsing
parser = reqparse.RequestParser()
parser.add_argument('product', type= str, help= "Please enter the product name")
parser.add_argument('quantity', type= int, help= "Please enter the quantity")

# Method to check the product available or not
def abort_if_product_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Product doesn't exist") # aborting the API call


# To fetch, update or delete a single product
class Cart(Resource):

    # get the product details by passing product id
    def get(self, product_id):
        abort_if_product_doesnt_exist(product_id) # Checking whether the product id exists
        return SHOPPING_CART[product_id]

    # updating the quantity of the product
    def put(self, product_id):
        args = parser.parse_args()
        SHOPPING_CART[product_id]['quantity'] = args['quantity']
        return SHOPPING_CART[product_id], 201 # return a JSON object, along with the return code 201 after updating

    # deleting the product
    def delete(self, product_id):
        abort_if_product_doesnt_exist(product_id) # Checking whether the product id exists
        del SHOPPING_CART[product_id]
        return '', 204 # return a JSON object, along with the return code 204 after deleting


# shows a list of all items in the cart and to add a new product to the cart
class CartList(Resource):

    # Fetch all the products in the cart.
    def get(self):
        return SHOPPING_CART  # returning a JSON object of all products

    # Adding a new product to the cart.
    def post(self):
        args = parser.parse_args()
        product_id = args['product'].lower()
        SHOPPING_CART[product_id] = {'product': args['product'], 'quantity': args['quantity']}
        return SHOPPING_CART[product_id], 201

# Api resource routing
api.add_resource(CartList, '/product') # Adding resources for routing for class CartList
api.add_resource(Cart, '/product/<product_id>') # Adding resources for routing for class Cart


if __name__ == '__main__':
    app.run(debug=True)