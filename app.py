import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from spotifyAPI import *

app = Flask(__name__)
model = pickle.load(open('models/gbt3.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inputs = [x for x in request.form.values()]
    track = inputs[-1]
    rem_features = inputs[0:10]
    client_id = '31f4904f55144d938e3a19f9f9636c4f'
    client_secret = '13521108cdee48538044a625a8da963c'
    spotify = SpotifyAPI(client_id, client_secret)
    my_dict = spotify.search({"track": track}, search_type="track")
    id = my_dict["tracks"]["items"][0]["id"]

    access_token = spotify.access_token
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    endpoint = "https://api.spotify.com/v1/audio-features"
    data = urlencode({"ids": id})

    lookup_url = f"{endpoint}?{data}"
    r = requests.get(lookup_url, headers=headers)
    abc = r.json()
    features = list(abc["audio_features"][0].values())[0:6]+list(abc["audio_features"][0].values())[7:11]+list(abc["audio_features"][0].values())[16:18]    
    #features = [np.array(features)]
    final_features = [np.array(rem_features + features)]
    prediction = model.predict(final_features)
    def fxprediction(prediction): 
        if prediction > 0.5:                
            return(" user will skip the song")
        else:
            return("user will not skip the song")

    return render_template('index.html', prediction_text='{}'.format(fxprediction(prediction)))

if(__name__ == "__main__"):
    app.run(debug = True)