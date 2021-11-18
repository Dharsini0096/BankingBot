from flask import Flask, request, jsonify

from model.bot2 import chatwithbot, getbankname

app = Flask(__name__)

@app.route('/chat', methods=['GET', 'POST'])
def chatBot():
    chatInput = request.form.get('chatInput')
    bankname = request.form.get("bankname")
    getbankname(bankname)
    return jsonify(chatBotReply=chatwithbot(chatInput))


if __name__ == '__main__':
    #app.run(host="192.168.43.90", debug=True)
    app.run(debug=True)
