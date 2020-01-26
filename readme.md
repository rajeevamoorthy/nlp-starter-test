
# NLP model for Disaster Prediction

This includes all parts of the assignment in one package. In a production system these should be separated into multiple packages to promote modularity with clear contracts between the services.

The codebase is containerised via docker which also exposes the webservice.

The following are the major functionalities:
- sqlite DB
    - script to decode zipped db file
- Train Library:
    - Classifier pipelinethat creates a trained model
    - Includes script to auto execute training during CI/CD pipeline
- Predict library:
    - a simple utility that takes a text input and a trained classifier model and returns prediction ['Relevant', 'Not Relevant']
- Predict App:
    - A web service that exposes Predict API service as a REST endpoint

## Standard best practices:
- This readme file for documentation.
- Use of virtual env and a requirements.txt to encapsulate dependencies
- Use of configuration files to standardize git (source control), coverage (testing), Linting (code quality), CI (SVN specific) etc.
- All source files to have inline documentation (to support some form of auto-doc library)
- All code to have unit testing (check coverage) and sufficient Integration testing.
- Appropriate use of exception management in source code.
- Common functionality to be extracted into a library function.
- Use Config files for app specific data to avoid hardcoding info in the source.

## Script to convert sqlite db from base64 to binary

    python sqlite_decode_script.py

Production Note: DB should be a stable backend and this script is present to demonstrate Pythonic way of extracting archives and file management.

## Utils
NLPPipeline implements a few commonly used functionality that both the train and predict models can rely on. There shall be no code dublication

## Training
- Training uses a rudimentary NLP classifier. Focus is on standardising the pipeline and not on the NLP model.
    - Read data
    - Prepare data (clean etc)
    - Feature extraction (model specific)
    - Test / train split
    - Fitting a model
    - Validating model (Precition/Recall)
    - Model persistence (if validation succeeds)

- The training can be triggered via a global script /nlp_aassignment/training_script.py which is also used in CI/CD automation pipeline
- Trained model is stored as a pickled file in local FS.

Production Note: Trained model should be dumped into some sensible backend service like S3 with appropriate access rights. Model replacement should have some form of backup / update policy.

## Predict Library
Uses an trained model from persistent storage and providesn an appropriate query API.
- input text string
- output Prediction as a json.
Notice the reuse of the Pipeline library to prepare the input text.

## Predict API
The webservice exposes the Predict Library functionality via a REST endpoint. Start the webservice via:
    docker-compose up --build

- Webserver auto executes when the container loads. Access via
    http://localhost:8080/predict?text=example

- API Specs are provided and is exposed using swagger ui.
    http://localhost:8080/swagger

Production Note: To use some form of Auto doc library to generate the API Specs.

## CI/CD
Using Gitlab script for CI.


## Run on Google cloud

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run/?git_repo=https://github.com/rajeevamoorthy/nlp-starter-test.git)

Unfortnately, there is some conflict with deployment to Google cloud run. Seemingly, some updates are needed in the docker script.
