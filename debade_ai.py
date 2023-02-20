from flask import Flask, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from flask_socketio import SocketIO

app = Flask(__name__)
sockio = SocketIO(app)

# create a thread decorated function to start a socket server
def start_socket_server():
    HOST = "127.0.0.1"
    PORT = 5001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)

def get_opinion_on(argument, persona):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(f'{persona}: According to my opinion regarding the {argument},', return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# create a flask endpoint that receives one persona and a topic and returns a the generated text to the repsonse
@app.route('/persona', methods=['GET', 'POST'])
def persona():
    method = request.method
    if method == 'GET':
        persona = request.args.get('persona')
        topic = request.args.get('topic')
    else:
        data = request.get_json()
        persona = data['persona']
        topic = data['topic']
    return get_opinion_on(topic, persona)

@app.route('/battle/<persona>/<topic>', methods=['PUT'])
def battle(persona, topic):
    """
    This endpoint receives two personas and a topic and returns a battle between the two personas
    The battle should contain all information about a socket server endpoint that the client can connect
    to receive the battle in real time
    """
    return get_opinion_on(topic, persona)


@sockio.on('message')
def handle_message(data):
    print('received message: ' + data)

if __name__ == '__main__':
    # start the api server
    sockio.run(app)

# create a sample request to the endpoint using curl in the terminal using curl
# curl -X GET http://localhost:5000/persona?persona=Epictetus&topic=death

# crreat a sample request to the endpoint using curl in the terminal
# curl -X POST http://localhost:5000/persona -d "{persona: 'Epictetus', topic: 'philosophy'}"