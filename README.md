# DisasterPipeline
This repo contains code and logic in order to create a prediction on a test-message in a disastrous situation to classify the label of this message, which should help in targetting the important messages.

This repo consists of the following parts:
- ETL: Which loads + cleans the CSV-datasets (see data-folder) and stores them in a sqlite-database
- Model-Creation: Trains a Random-Forest model on this data and stores an output of the model as a pickle file
- App: Sets-up a html-page with some visuals related to the data, besides it offers an input option to see the prediction on any message you enter
 
## Installation
In order to run all the code in this repo you need at least need to have a python installation and the python packages `nltk`, `sklearn`, `sqlalchemy (version == 1.4.46)`

Besides, you need to have donwloaded the following nltk content: `['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']`

##### Example
```
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
```

## Usage
Once initiated the game can be started with the ".play()" function
Also see the "3-in-a-row game example.ipynb" for an example of how to play.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!


## Fork the Project
- Create your Feature Branch (git checkout -b feature/AmazingFeature)
- Commit your Changes (git commit -m 'Add some AmazingFeature')
- Push to the Branch (git push origin feature/AmazingFeature)
- Open a Pull Request


## License
Distributed under the MIT License. See LICENSE.txt for more information.


## Contact
Joep Smulders - (https://www.linkedin.com/in/joep-smulders-200203117/) - smulders.jmm@gmail.com

Project Link: (https://github.com/jmmsmulders/ThreeInARow)
