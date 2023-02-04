# DisasterPipeline
This repo contains code and logic in order to create a prediction on a test-message in a disastrous situation to classify the label of this message, which should help in targetting the important messages.

This repo consists of the following parts:
- ETL: Which loads + cleans the CSV-datasets (see data-folder) and stores them in a sqlite-database
- Model-Creation: Trains a Random-Forest model on the cleaned dataset and stores an output of the model as a pickle file
- Web-App: Sets-up a html-page with some visuals related to the data, besides it offers an input option to see the prediction on any message you enter
 
## Installation
In order to run all the code in this repo you need at least need to have a python installation and the python packages `nltk`, `sklearn`, `sqlalchemy (version == 1.4.46)`

Besides, you need to have donwloaded the following nltk content: `['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']`

##### Example
```
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
```

## Usage
There are 3 steps that could be run in this repo, first you need to clone this repo and navigate (within a terminal) to it's location, then the following processes could be run with the following commands
#### ETL
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
#### Model Creation
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
#### Web-App
First navigate to the app folder with `cd app` then run the following command to create the Web-App `python run.py`

This results in the creation of the following web-app, which you can access by following the http link in your terminal:
![image](https://user-images.githubusercontent.com/118716035/216780454-86b09948-6e7c-4710-8e1b-c2373f7a7005.png)

Where you can manual enter a message which will then be classified by the trained algorithm, which will look as follows:
![image](https://user-images.githubusercontent.com/118716035/216780510-32990b89-8715-4085-9109-dc04018bfffa.png)


## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

## Fork the Project
- Create your Feature Branch (git checkout -b feature/AmazingFeature)
- Commit your Changes (git commit -m 'Add some AmazingFeature')
- Push to the Branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## Contact
Joep Smulders - (https://www.linkedin.com/in/joep-smulders-200203117/) - smulders.jmm@gmail.com

Project Link: (https://github.com/jmmsmulders/ThreeInARow)
