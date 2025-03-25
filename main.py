import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.models import load_model
from torchvision import transforms
from PIL import Image
import pickle
import os
import base64
import sklearn
from datetime import datetime
import llm_helper

app = Flask(__name__)

randomForest_model = pickle.load(open('randomForest_model.pkl', 'rb'))
logisticR_model = pickle.load(open('logisticRegression_model.pkl', 'rb'))
decisionTree_model = pickle.load(open('decisionTree_model.pkl', 'rb'))

classes = ['Diabetic', 'Non-Diabetic']    

def findConv2dOutShape(hin,win,conv,pool=2):
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)

# Define Architecture For Retinopathy Model
class CNN_Retino(nn.Module):
    
    def __init__(self, params):
        
        super(CNN_Retino, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # CNN Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

params_model={
        "shape_in": (3,255,255), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.15,
        "num_classes": 2}

# Create instantiation of Network class
Retina_model = CNN_Retino(params_model)

# Load the model's state_dict
# model_state_dict = torch.load("./Retino_model_with_state.pt")
# Retina_model.load_state_dict(model_state_dict)

# Set model to evaluation mode
# Retina_model.eval()

# Define prediction function
def predict_retina(image_path):
    image = Image.open(image_path)

    image = transform(image).unsqueeze(0)
    print("-----------------Image transform success ----------")

    with torch.no_grad():
        output = Retina_model(image)
        # print("-----------------output shape: ", output.shape)
        print("-----------------output values: ", output)
    _, predicted = torch.max(output, 1)
    print("-----------------Predict returning : ",predicted.item())
    return predicted.item()

# Define Transformation for Retina
transform = transforms.Compose(
    [
        transforms.Resize((255,255)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)

@app.route('/retina_model')
def retina_page():
   return render_template('retina.html')


@app.route('/predict_from_retina', methods=['POST'])
def retina_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        print("File received:", file)
        class_index = predict_retina(file)
        print("-----------------class index :", class_index)
        # Assuming you have a list of class names
        class_name = classes[class_index]  # Make sure you define class_names
        print("------------------class name :", class_name)
        result = "Patient is " + class_name;

        # Save the uploaded image to a temporary directory
        upload_folder = '/static/uploads/'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)

        # Encode the image file to base64
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        return render_template('retina.html',result= result, image_path=image_path, encoded_image=encoded_image)
    except Exception as e:
        print("Prediction failed:", str(e))
        return jsonify({'error': 'Prediction failed'})


@app.route('/predict', methods=['POST'])
def predict():
  # Print the data type of each form value
#   for key, value in request.form.items():
#     print(f"Field: {key}, Value: {value}, Type: {type(value)}")
  int_features = [int(x) for x in request.form.values()]
  final_features = [np.array(int_features)]
  
  RFprediction = randomForest_model.predict(final_features)
  LRprediction = logisticR_model.predict(final_features)
  DTprediction = decisionTree_model.predict(final_features)
  
  # output = round(prediction[0], 2)
  output = [RFprediction[0], LRprediction[0], DTprediction[0]]
  # output[0] = RFprediction[0]
  # output[1] = LRprediction[0]
  # output[2] = DTprediction[0]
  
  print("RFOutput : ", output[0])
  print("LROutput : ", output[1])
  print("DTOutput : ", output[2])

  result = ["Negative", "Negative", "Negative"]
  for i in range(3):
    if output[i] == 0.0:
        result[i] = "Negative"
    else:
        result[i] = "Positive"

  # prediction_text = ['Random Forest_______: Patient is ', 
  #                    'Logistic Regression_: Patient is ', 
  #                    'Decision Tree_______: Patient is ']
  # for i in range(3):
  #   prediction_text[i] += result[i]
    
  return render_template('index.html',
    prediction_text1='Random Forest_______: Patient is {}'.format(result[0]), 
    prediction_text2='Logistic Regression : Patient is {}'.format(result[1]),
    prediction_text3='Decision Tree_______: Patient is {}'.format(result[2]))

@app.route('/chatbot')
def chatbot_page():
   return render_template('chatbot.html')

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    data = request.get_json()
    prompt = data.get('prompt')

    result = llm_helper.ask_query(prompt);
    response_text = result['answer']
    response_link = result['sources']

    # current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # dt = formatted_time
    # # Process the prompt and generate a response
    # response_text = f'You said: {prompt}\nD&T: {dt}'  # Example response text
    
    return jsonify({'response_ans': response_text, 'response_src' : response_link})

@app.route('/')
def home():
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=80)
