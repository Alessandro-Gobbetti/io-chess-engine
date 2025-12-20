#!/bin/bash
# get project dir
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")/data
echo "Downloading chess evaluations to $PROJECT_DIR/chess-evaluations.zip"
curl -L -o $PROJECT_DIR/chess-evaluations.zip\
  https://www.kaggle.com/api/v1/datasets/download/ronakbadhe/chess-evaluations
unzip $PROJECT_DIR/chess-evaluations.zip -d $PROJECT_DIR
rm $PROJECT_DIR/chess-evaluations.zip